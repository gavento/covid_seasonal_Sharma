import argparse
import csv
import json

import arviz
import numpy as np
import pandas as pd


def st(d, ci=0.95, dec=3, short=False):
    d = np.array(d)
    c = np.quantile(d, [0.5 - ci / 2, 0.5, 0.5 + ci / 2])
    if not ci:
        return f"{np.mean(d):.{dec}f} (sd={np.std(d):.{dec}f})"
    if short:
        return f"{np.mean(d):.{dec}f} ({c[0]:.{dec}f}..{c[2]:.{dec}f})"
    return f"{np.mean(d):.{dec}f} (sd={np.std(d):.{dec}f}, 95%CI {c[0]:.{dec}f} .. {c[2]:.{dec}f})"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_json")
    parser.add_argument("-w", "--write_csv", action="store_true")
    parser.add_argument("-p", "--plot_dists", action="store_true")
    args = parser.parse_args()

    with open(args.summary_json) as f:
        s = json.load(f)
    f = arviz.from_netcdf(args.summary_json.replace("_summary.json", "_full.netcdf"))
    chains = len(f.posterior.chain)
    tot_eff = 100 * (1 - np.exp(-np.sum(f.posterior.alpha_i, axis=-1)))
    rtw_log = np.array(np.log(f.posterior.Rt_walk))
    csv_cols = []
    print(
        f"""Loaded {args.summary_json}

  experiment: {s['exp_tag']} {s['exp_config']}, chains: {chains}x {len(s['warmup']['num_steps']) // chains} + {len(s['sample']['num_steps']) // chains} samples
  model:      {s['model_name']} {s['model_kwargs']}, config={s['model_config_name']!r}

  rhat={s['rhat']['lower']:.3f}-{s['rhat']['upper']:.3f}, divergences={s['divergences']}, accept_prob={st(s['sample']['mean_accept_prob'])}, total_runtime={s['total_runtime']:.2f} s
  basic_R={st(f.posterior.basic_R)}, total effect={st(tot_eff)}
  log(Rt_walk)={st(rtw_log)}, Rt_walk^2 in logspace: {st(rtw_log**2)}"""
    )

    def fl(x):
        return np.array(x).flatten()

    csv_cols.append(
        pd.Series(fl(f.posterior.basic_R.mean(axis=-1)), name="basic_R")
    )
    csv_cols.append(
        pd.Series(fl(f.posterior.basic_R.std(axis=-1)), name="basic_R_sd")
    )
    csv_cols.append(pd.Series(fl(tot_eff), name="total_effect"))

    if "seasonality_beta1" in f.posterior:
        b1 = f.posterior.seasonality_beta1
        print(
            f"  seasonality_beta1={st(b1)}, R0(Jan 1) / R0(July 1) = {st((1 + b1) / (1-b1))}\n"
            f'  equivalent NPI effect of "summer July 1" (vs "Jan 1") = {st(100*(1 - (1 - b1) / (1 + b1)))}'
        )
        csv_cols.append(pd.Series(fl(b1), name="beta_1"))
        csv_cols.append(
            pd.Series(
                fl(f.posterior.basic_R.mean(axis=-1))
                / fl(f.posterior.seasonality_multiplier[:, :, 0]),
                name="no_seasonality_basic_R",
            )
        )

    if "basic_R_prior_mean" in f.posterior:
        brh_m, brh_s = f.posterior.basic_R_prior_mean, f.posterior.basic_R_prior_scale
        print(f"basic_R hyperprior:  mean={st(brh_m)}  scale={st(brh_s)}")
        csv_cols.append(pd.Series(fl(brh_m), name="basic_R_prior_mean"))
        csv_cols.append(pd.Series(fl(brh_s), name="basic_R_prior_scale"))

    if "seasonality_max_R_day" in f.posterior:
        mRd = f.posterior.seasonality_max_R_day
        print(f"seasonality_max_R_day: {st(mRd)}")
        csv_cols.append(pd.Series(fl(mRd), name="seasonality_max_R_day"))

    efs = [st(100 * (1 - np.exp(-d)), dec=2, short=True) for d in f.posterior.alpha_i.T]
    print("\n  effects(95% CI):")
    while efs:
        print("    ", ", ".join(efs[:5]))
        efs = efs[5:]
    print()

    if args.write_csv:
        for i, d in enumerate(f.posterior.alpha_i.T):
            csv_cols.append(pd.Series(fl(d), name=f"alpha/{s['cm_names'][i]}"))
        assert all(len(c)==len(csv_cols[0]) for c in csv_cols)
        df = pd.DataFrame({c.name: c for c in csv_cols})
        df.to_csv(
            args.summary_json.replace("_summary.json", "_stats.csv.xz"), index=False
        )

    if args.plot_dists:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Dists for {s['exp_tag']}\n{s['exp_config']}")

        axes[0, 0].set_title(
            f"Rt_walk in logspace: {st(rtw_log, ci=False)}\nRt_walk^2 in logspace: {st(rtw_log**2, ci=False)}"
        )
        axes[0, 0].set_xlim([-1.0, 1.0])
        sns.histplot(rtw_log.flatten(), ax=axes[0, 0], bins=80, binrange=[-1.0, 1.0])

        axes[1, 0].set_title(f"basic_R: {st(f.posterior.basic_R, ci=False)}")
        axes[1, 0].set_xlim([0.5, 3.0])
        sns.histplot(
            np.array(f.posterior.basic_R).flatten(),
            ax=axes[1, 0],
            bins=80,
            binrange=[0.5, 3.0],
        )

        axes[0, 1].set_title(f"total NPI effect: {st(tot_eff, short=True)}")
        axes[0, 1].set_xlim([50, 90])
        sns.histplot(
            np.array(tot_eff).flatten(), ax=axes[0, 1], bins=80, binrange=[50, 90]
        )

        axes[1, 1].set_xlim([-0.1, 0.6])
        axes[1, 1].set_title(f"seasonality beta_1: NA")
        if "seasonality_beta1" in f.posterior:
            axes[1, 1].set_title(f"seasonality beta_1: {st(b1, short=True)}")
            sns.histplot(
                np.array(b1).flatten(), ax=axes[1, 1], bins=80, binrange=[-0.1, 0.6]
            )

        plt.tight_layout()
        plt.savefig(args.summary_json.replace("_summary.json", "_dists.pdf"))
