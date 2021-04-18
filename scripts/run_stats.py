import argparse
import csv
import json

import arviz
import numpy as np


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
    parser.add_argument("-w", "--write_seasonality", action="store_true")
    parser.add_argument("-p", "--plot_dists", action="store_true")
    args = parser.parse_args()

    with open(args.summary_json) as f:
        s = json.load(f)
    f = arviz.from_netcdf(args.summary_json.replace("_summary.json", "_full.netcdf"))
    chains = len(f.posterior.chain)
    tot_eff = 100 * (1 - np.exp(-np.sum(f.posterior.alpha_i, axis=-1)))
    rtw_log = np.array(np.log(f.posterior.Rt_walk))
    print(
        f"""Loaded {args.summary_json}

  experiment: {s['exp_tag']} {s['exp_config']}, chains: {chains}x {len(s['warmup']['num_steps']) // chains} + {len(s['sample']['num_steps']) // chains} samples
  model:      {s['model_name']} {s['model_kwargs']}, config={s['model_config_name']!r}

  rhat={s['rhat']['lower']:.3f}-{s['rhat']['upper']:.3f}, divergences={s['divergences']}, accept_prob={st(s['sample']['mean_accept_prob'])}, total_runtime={s['total_runtime']:.2f} s
  basic_R={st(f.posterior.basic_R)}, total effect={st(tot_eff)}
  log(Rt_walk)={st(rtw_log)}, Rt_walk^2 in logspace: {st(rtw_log**2)}"""
    )
    if "seasonality_beta1" in f.posterior:
        b1 = f.posterior.seasonality_beta1
        print(
            f"  seasonality_beta1={st(b1)}, R0(Jan 1) / R0(July 1) = {st((1 + b1) / (1-b1))}\n"
            f'  equivalent NPI effect of "summer July 1" (vs "Jan 1") = {st(100*(1 - (1 - b1) / (1 + b1)))}'
        )
        if args.write_seasonality:
            with open(
                args.summary_json.replace("_summary.json", "_beta1.csv"), "wt"
            ) as cf:
                cfw = csv.writer(cf)
                cfw.writerow(
                    [f"Sharma {s['exp_tag']} {s['model_config_name']} {s['exp_config']}\n"]
                )
                cfw.writerows([[str(b)] for b in np.array(b1).flatten()])

    if "basic_R_prior_mean" in f.posterior:
        brh_m, brh_s = f.posterior.basic_R_prior_mean, f.posterior.basic_R_prior_scale
        print(f"basic_R hyperprior:  mean={st(brh_m)}  scale={st(brh_s)}")

    efs = [st(100 * (1 - np.exp(-d)), dec=2, short=True) for d in f.posterior.alpha_i.T]
    print("\n  effects(95% CI):")
    while efs:
        print("    ", ", ".join(efs[:5]))
        efs = efs[5:]
    print()

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
