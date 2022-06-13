"""
Extra models - interactions and higher-order seasonality
"""
import jax.scipy.signal
import jax.numpy as jnp
import jax
import numpyro

from epimodel.models.model_build_utils import *

@numpyro.handlers.reparam(config={"seasonality_phases_tail": numpyro.infer.reparam.CircularReparam()})
def seasonality_fourier_model(
    data,
    ep,
    intervention_prior=None,
    basic_R_prior=None,
    r_walk_noise_scale_prior=0.15,
    r_walk_period=7,
    n_days_seeding=7,
    seeding_scale=3.0,
    infection_noise_scale=5.0,
    output_noise_scale_prior=5.0,
    fourier_degree=0,
    **kwargs,
):
    """
    Main model with cosine seasonality.

    :param data: PreprocessedData object
    :param ep: EpidemiologicalParameters object
    :param intervention_prior: intervention prior dict
    :param basic_R_prior: basic r prior dict
    :param r_walk_noise_scale_prior: scale of random walk noise scale prior
    :param r_walk_period: period of random walk
    :param n_days_seeding: number of days of seeding
    :param seeding_scale: scale of seeded infection prior
    :param infection_noise_scale: scale of infection noise
    :param output_noise_scale_prior: output noise scale prior
    :param kwargs: additional kwargs (not used, but maintain function signature)
    """
    for k in kwargs.keys():
        print(f"{k} is not being used")

    assert fourier_degree >= 1
    periods=[365.0,]
    for i in range(fourier_degree - 1):
        periods.append(periods[-1] / 2.0)
    periods=jnp.array(periods)
    assert periods.shape == (fourier_degree, )

    # First, compute R.
    # sample intervention effects from their priors.
    # mean intervention effects
    alpha_i = sample_intervention_effects(data.nCMs, intervention_prior)
    # transmission reduction
    cm_reduction = jnp.sum(data.active_cms * alpha_i.reshape((1, data.nCMs, 1)), axis=1)

    # sample basic R
    # NB: SEASONALITY interpretation: this is the R0 on the first day already including seasonality
    # Therefore it is comparable to basic_R without seasonality
    basic_R = sample_basic_R(data.nRs, basic_R_prior)

    # number of 'noise points'
    # -1 since no change for the first 2 weeks.
    nNP = int(data.nDs / r_walk_period) - 1

    r_walk_noise_scale = numpyro.sample(
        "r_walk_noise_scale", dist.HalfNormal(scale=r_walk_noise_scale_prior)
    )

    if fourier_degree > 1:
        seasonality_phases_tail = numpyro.sample(
            "seasonality_phases_tail",
            dist.VonMises(jnp.zeros(fourier_degree - 1), 0.001)) # Uninformaive prior
        seasonality_max_R_day_vec = numpyro.deterministic(
            "seasonality_max_R_day_vec",
            jnp.concatenate([jnp.array([1.0]), seasonality_phases_tail / 2 / jnp.pi * periods[1:]])
        )
    else:
        seasonality_max_R_day_vec = numpyro.deterministic(
            "seasonality_max_R_day_vec",
            jnp.array([1.0])
        )

    seasonality_beta1 = numpyro.sample(
        "seasonality_beta1", dist.Uniform(jnp.zeros(fourier_degree), 0.95)
    )

    seasonality_multiplier = numpyro.deterministic(
        "seasonality_multiplier",
        1.0
        + jnp.sum(seasonality_beta1
        * jnp.cos(
            (data.Ds_day_of_year.reshape((-1, 1)) - seasonality_max_R_day_vec.reshape((1, fourier_degree)))
            / periods
            * 2.0
            * jnp.pi
        ), axis=1),
    ).reshape((1, -1))

    # rescaling variables by 10 for better NUTS adaptation
    r_walk_noise = numpyro.sample(
        "r_walk_noise",
        dist.Normal(loc=jnp.zeros((data.nRs, nNP)), scale=1.0 / 10),
    )

    # only apply the noise every "r_walk_period" - to get full noise, repeat
    expanded_r_walk_noise = jnp.repeat(
        r_walk_noise_scale * 10.0 * jnp.cumsum(r_walk_noise, axis=-1),
        r_walk_period,
        axis=-1,
    )[: data.nRs, : (data.nDs - 2 * r_walk_period)]
    # except that we assume no noise for the first 3 weeks
    full_log_Rt_noise = jnp.zeros((data.nRs, data.nDs))
    full_log_Rt_noise = full_log_Rt_noise.at[:, 2 * r_walk_period :].set(expanded_r_walk_noise)

    # NB: basic_R is R0(t=0) INCLUDING seasonality effect (for comparability with non-seasonal model),
    # so we need to divide by the initial seasonality first
    Rt = numpyro.deterministic(
        "Rt",
        jnp.exp(
            jnp.log(basic_R.reshape((data.nRs, 1))) + full_log_Rt_noise - cm_reduction
        )
        * seasonality_multiplier
        / seasonality_multiplier[:, :1],  # First day multipliers for every region
    )

    # collect variables in the numpyro trace
    numpyro.deterministic("Rt_walk", jnp.exp(full_log_Rt_noise))
    numpyro.deterministic(
        "Rt_cm", jnp.exp(jnp.log(basic_R.reshape((data.nRs, 1))) - cm_reduction)
    )
    # Rt before applying cm_reductions
    numpyro.deterministic(
        "Rt0",
        jnp.exp(jnp.log(basic_R.reshape((data.nRs, 1))) + full_log_Rt_noise)
        * seasonality_multiplier
        / seasonality_multiplier[:, :1],
    )

    # Infection Model
    seeding_padding = n_days_seeding
    total_padding = ep.GIv.size - 1

    # note; seeding is also rescaled
    init_infections, total_infections_placeholder = seed_infections(
        seeding_scale, data.nRs, data.nDs, seeding_padding, total_padding
    )
    discrete_renewal_transition = get_discrete_renewal_transition(ep)

    # we need to transpose R because jax.lax.scan scans over the first dimension.
    # We want to scan over time instead of regions!
    _, infections = jax.lax.scan(
        discrete_renewal_transition,
        init_infections,
        Rt.T,
    )

    # corrupt infections with additive noise, adding robustness at small case and death
    # counts
    infection_noise = numpyro.sample(
        "infection_noise",
        dist.Normal(loc=0, scale=0.1 * jnp.ones((data.nRs, data.nDs))),
    )
    # enforce positivity!
    infections = jax.nn.softplus(
        infections + (infection_noise_scale * (10.0 * infection_noise.T))
    )

    total_infections = total_infections_placeholder.at[:, :seeding_padding].set(init_infections[:, -seeding_padding:])
    total_infections = numpyro.deterministic(
        "total_infections",
        total_infections.at[:, seeding_padding:].set(infections.T),
    )

    # Time constant case fatality rate (ascertainment rate assumed to be 1
    # throughout the whole period).
    cfr = numpyro.sample("cfr", dist.Uniform(low=1e-3, high=jnp.ones((data.nRs, 1))))

    future_cases_t = numpyro.deterministic("future_cases_t", total_infections)
    future_deaths_t = numpyro.deterministic(
        "future_deaths_t", jnp.multiply(total_infections, cfr)
    )

    # collect expected cases and deaths
    expected_cases = numpyro.deterministic(
        "expected_cases",
        jax.scipy.signal.convolve2d(future_cases_t, ep.DPC, mode="full")[
            :, seeding_padding : seeding_padding + data.nDs
        ],
    )

    expected_deaths = numpyro.deterministic(
        "expected_deaths",
        jax.scipy.signal.convolve2d(future_deaths_t, ep.DPD, mode="full")[
            :, seeding_padding : seeding_padding + data.nDs
        ],
    )

    # country specific psi cases and deaths.
    # We will use the 'RC' matrix to pull the correct local area value.
    psi_cases = numpyro.sample(
        "psi_cases",
        dist.HalfNormal(scale=output_noise_scale_prior * jnp.ones(len(data.unique_Cs))),
    )
    psi_deaths = numpyro.sample(
        "psi_deaths",
        dist.HalfNormal(scale=output_noise_scale_prior * jnp.ones(len(data.unique_Cs))),
    )

    # use the per country psi_cases and psi_deaths and form a nRs x nDs array
    # to use for the output distribution.
    cases_conc = (
        (data.RC_mat @ psi_cases).reshape((data.nRs, 1)).repeat(data.nDs, axis=-1)
    )
    deaths_conc = (
        (data.RC_mat @ psi_deaths).reshape((data.nRs, 1)).repeat(data.nDs, axis=-1)
    )

    with numpyro.handlers.mask(mask=jnp.logical_not(data.new_cases.mask)):
        numpyro.sample(
            "observed_cases",
            dist.GammaPoisson(
                concentration=cases_conc,
                rate=cases_conc / expected_cases,
            ),
            obs=data.new_cases.data,
        )

    with numpyro.handlers.mask(mask=jnp.logical_not(data.new_deaths.mask)):
        numpyro.sample(
            "observed_deaths",
            dist.GammaPoisson(
                concentration=deaths_conc,
                rate=deaths_conc / expected_deaths,
            ),
            obs=data.new_deaths.data,
        )

seasonality_fourier_model.__name__ = "seasonality_fourier_model"


def seasonality_interactions_model(
    data,
    ep,
    intervention_prior=None,
    basic_R_prior=None,
    r_walk_noise_scale_prior=0.15,
    r_walk_period=7,
    n_days_seeding=7,
    seeding_scale=3.0,
    infection_noise_scale=5.0,
    output_noise_scale_prior=5.0,
    seasonality_prior="uniform",
    max_R_day_prior=None,
    different_seasonality=False,
    local_seasonality_sd=0.1,
    interactions=None,
    interactions_sd=1.0,
    **kwargs,
):
    """
    Main model with cosine seasonality.

    :param data: PreprocessedData object
    :param ep: EpidemiologicalParameters object
    :param intervention_prior: intervention prior dict
    :param basic_R_prior: basic r prior dict
    :param r_walk_noise_scale_prior: scale of random walk noise scale prior
    :param r_walk_period: period of random walk
    :param n_days_seeding: number of days of seeding
    :param seeding_scale: scale of seeded infection prior
    :param infection_noise_scale: scale of infection noise
    :param output_noise_scale_prior: output noise scale prior
    :param seasonality_prior: type of prior to use for seasonality scale (just "uniform")
    :param max_R_day_prior: prior dict for day of maximum R from seasonality
    :param kwargs: additional kwargs (not used, but maintain function signature)
    """
    assert interactions is not None
    for k in kwargs.keys():
        print(f"{k} is not being used")

    # First, compute R.
    # sample intervention effects from their priors.
    # mean intervention effects
    alpha_i = sample_intervention_effects(data.nCMs, intervention_prior)
    # transmission reduction
    cm_reduction = jnp.sum(data.active_cms * alpha_i.reshape((1, data.nCMs, 1)), axis=1)

    # sample basic R
    # NB: SEASONALITY interpretation: this is the R0 on the first day already including seasonality
    # Therefore it is comparable to basic_R without seasonality
    basic_R = sample_basic_R(data.nRs, basic_R_prior)

    # number of 'noise points'
    # -1 since no change for the first 2 weeks.
    nNP = int(data.nDs / r_walk_period) - 1

    r_walk_noise_scale = numpyro.sample(
        "r_walk_noise_scale", dist.HalfNormal(scale=r_walk_noise_scale_prior)
    )

    if max_R_day_prior["type"] == "normal":
        seasonality_max_R_day = numpyro.sample(
            "seasonality_max_R_day",
            dist.Normal(max_R_day_prior["mean"], max_R_day_prior["scale"]),
        )
    elif max_R_day_prior["type"] == "fixed":
        seasonality_max_R_day = numpyro.deterministic(
            "seasonality_max_R_day",
            jnp.array(max_R_day_prior["value"], dtype=jnp.float32),
        )
    else:
        raise Exception(f"Invalid max_R_day_prior")

    if seasonality_prior == "uniform":
        seasonality_beta1 = numpyro.sample(
            "seasonality_beta1", dist.Uniform(-0.95, 0.95)
        )
    else:
        raise Exception("Invalid seasonality_prior")

    seasonality_beta1_bc = seasonality_beta1 * jnp.ones(data.nCs)
    if different_seasonality:
        seasonality_local_beta1 = numpyro.sample(
            "seasonality_local_beta1",
            dist.Normal(seasonality_beta1_bc, scale=local_seasonality_sd),
        )
    else:
        seasonality_local_beta1 = numpyro.deterministic(
            "seasonality_local_beta1", seasonality_beta1_bc
        )

    seasonality_region_beta1 = seasonality_local_beta1 @ data.RC_mat.T
    seasonality_multiplier = numpyro.deterministic(
        "seasonality_multiplier",
        1.0
        + seasonality_region_beta1.reshape((data.nRs, 1))
        * jnp.cos(
            (data.Ds_day_of_year - seasonality_max_R_day).reshape((1, data.nDs))
            / 365.0
            * 2.0
            * jnp.pi
        ),
    )

    seasonality_multiplier_full = numpyro.deterministic(
        "seasonality_multiplier_full",
        1.0
        + jnp.cos(
            (data.Ds_day_of_year - seasonality_max_R_day).reshape((1, data.nDs))
            / 365.0
            * 2.0
            * jnp.pi
        ),
    )

    # Interaction weights - sample with the same prior or as normals
    #alpha_int_i = sample_intervention_effects(data.nCMs, intervention_prior, name="alpha_int_i")
    alpha_int_i = numpyro.sample("alpha_int_i", dist.Normal(jnp.zeros(data.nCMs), interactions_sd))

    # transmission reduction from interactions
    assert seasonality_multiplier.shape == (data.nRs, data.nDs)
    assert seasonality_multiplier_full.shape == (1, data.nDs)
    assert data.active_cms.shape == (data.nRs, data.nCMs, data.nDs)
    if interactions == "with_full":
        print("Interactions with full-amplittude (0..2) cosine wave")
        cm_reduction_int = jnp.sum(data.active_cms * alpha_int_i.reshape((1, data.nCMs, 1)) * seasonality_multiplier_full.reshape(1, 1, data.nDs), axis=1)
    else:
        print("Interactions with seasonality-amplitude cosine wave")
        cm_reduction_int = jnp.sum(data.active_cms * alpha_int_i.reshape((1, data.nCMs, 1)) * seasonality_multiplier.reshape(data.nRs, 1, data.nDs), axis=1)

    # rescaling variables by 10 for better NUTS adaptation
    r_walk_noise = numpyro.sample(
        "r_walk_noise",
        dist.Normal(loc=jnp.zeros((data.nRs, nNP)), scale=1.0 / 10),
    )

    # only apply the noise every "r_walk_period" - to get full noise, repeat
    expanded_r_walk_noise = jnp.repeat(
        r_walk_noise_scale * 10.0 * jnp.cumsum(r_walk_noise, axis=-1),
        r_walk_period,
        axis=-1,
    )[: data.nRs, : (data.nDs - 2 * r_walk_period)]
    # except that we assume no noise for the first 3 weeks
    full_log_Rt_noise = jnp.zeros((data.nRs, data.nDs))
    full_log_Rt_noise = full_log_Rt_noise.at[:, 2 * r_walk_period :].set(expanded_r_walk_noise)

    # NB: basic_R is R0(t=0) INCLUDING seasonality effect (for comparability with non-seasonal model),
    # so we need to divide by the initial seasonality first
    Rt = numpyro.deterministic(
        "Rt",
        jnp.exp(
            jnp.log(basic_R.reshape((data.nRs, 1))) + full_log_Rt_noise - cm_reduction - cm_reduction_int
        )
        * seasonality_multiplier
        / seasonality_multiplier[:, :1],  # First day multipliers for every region
    )

    # collect variables in the numpyro trace
    numpyro.deterministic("Rt_walk", jnp.exp(full_log_Rt_noise))
    numpyro.deterministic(
        "Rt_cm", jnp.exp(jnp.log(basic_R.reshape((data.nRs, 1))) - cm_reduction - cm_reduction_int)
    )
    # Rt before applying cm_reductions
    numpyro.deterministic(
        "Rt0",
        jnp.exp(jnp.log(basic_R.reshape((data.nRs, 1))) + full_log_Rt_noise)
        * seasonality_multiplier
        / seasonality_multiplier[:, :1],
    )

    # Infection Model
    seeding_padding = n_days_seeding
    total_padding = ep.GIv.size - 1

    # note; seeding is also rescaled
    init_infections, total_infections_placeholder = seed_infections(
        seeding_scale, data.nRs, data.nDs, seeding_padding, total_padding
    )
    discrete_renewal_transition = get_discrete_renewal_transition(ep)

    # we need to transpose R because jax.lax.scan scans over the first dimension.
    # We want to scan over time instead of regions!
    _, infections = jax.lax.scan(
        discrete_renewal_transition,
        init_infections,
        Rt.T,
    )

    # corrupt infections with additive noise, adding robustness at small case and death
    # counts
    infection_noise = numpyro.sample(
        "infection_noise",
        dist.Normal(loc=0, scale=0.1 * jnp.ones((data.nRs, data.nDs))),
    )
    # enforce positivity!
    infections = jax.nn.softplus(
        infections + (infection_noise_scale * (10.0 * infection_noise.T))
    )


    total_infections = total_infections_placeholder.at[:, :seeding_padding].set(init_infections[:, -seeding_padding:])
    total_infections = numpyro.deterministic(
        "total_infections",
        total_infections.at[:, seeding_padding:].set(infections.T),
    )

    # Time constant case fatality rate (ascertainment rate assumed to be 1
    # throughout the whole period).
    cfr = numpyro.sample("cfr", dist.Uniform(low=1e-3, high=jnp.ones((data.nRs, 1))))

    future_cases_t = numpyro.deterministic("future_cases_t", total_infections)
    future_deaths_t = numpyro.deterministic(
        "future_deaths_t", jnp.multiply(total_infections, cfr)
    )

    # collect expected cases and deaths
    expected_cases = numpyro.deterministic(
        "expected_cases",
        jax.scipy.signal.convolve2d(future_cases_t, ep.DPC, mode="full")[
            :, seeding_padding : seeding_padding + data.nDs
        ],
    )

    expected_deaths = numpyro.deterministic(
        "expected_deaths",
        jax.scipy.signal.convolve2d(future_deaths_t, ep.DPD, mode="full")[
            :, seeding_padding : seeding_padding + data.nDs
        ],
    )

    # country specific psi cases and deaths.
    # We will use the 'RC' matrix to pull the correct local area value.
    psi_cases = numpyro.sample(
        "psi_cases",
        dist.HalfNormal(scale=output_noise_scale_prior * jnp.ones(len(data.unique_Cs))),
    )
    psi_deaths = numpyro.sample(
        "psi_deaths",
        dist.HalfNormal(scale=output_noise_scale_prior * jnp.ones(len(data.unique_Cs))),
    )

    # use the per country psi_cases and psi_deaths and form a nRs x nDs array
    # to use for the output distribution.
    cases_conc = (
        (data.RC_mat @ psi_cases).reshape((data.nRs, 1)).repeat(data.nDs, axis=-1)
    )
    deaths_conc = (
        (data.RC_mat @ psi_deaths).reshape((data.nRs, 1)).repeat(data.nDs, axis=-1)
    )

    with numpyro.handlers.mask(mask=jnp.logical_not(data.new_cases.mask)):
        numpyro.sample(
            "observed_cases",
            dist.GammaPoisson(
                concentration=cases_conc,
                rate=cases_conc / expected_cases,
            ),
            obs=data.new_cases.data,
        )

    with numpyro.handlers.mask(mask=jnp.logical_not(data.new_deaths.mask)):
        numpyro.sample(
            "observed_deaths",
            dist.GammaPoisson(
                concentration=deaths_conc,
                rate=deaths_conc / expected_deaths,
            ),
            obs=data.new_deaths.data,
        )
