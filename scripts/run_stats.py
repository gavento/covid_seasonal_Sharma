import numpy as np
import arviz
import json
import argparse


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
    parser.add_argument("-w", "--write_seasonality", type=str)
    parser.add_argument("-p", "--plot_dists", action="store_true")
    args = parser.parse_args()

    with open(args.summary_json) as f:
        s = json.load(f)
    f = arviz.from_netcdf(args.summary_json.replace("_summary.json", "_full.netcdf"))
    chains = len(f.posterior.chain)
    tot_eff = 100 * (1 - np.exp(-np.sum(f.posterior.alpha_i, axis=-1)))
    print(
        f"""Loaded {args.summary_json}

  experiment: {s['exp_tag']} {s['exp_config']}, chains: {chains}x {len(s['warmup']['num_steps']) // chains} + {len(s['sample']['num_steps']) // chains} samples
  model:      {s['model_name']} {s['model_kwargs']}, config={s['model_config_name']!r}

  rhat={s['rhat']['lower']:.3f}-{s['rhat']['upper']:.3f}, divergences={s['divergences']}, accept_prob={st(s['sample']['mean_accept_prob'])}, total_runtime={s['total_runtime']:.2f} s
  basic_R={st(f.posterior.basic_R)}, total effect={st(tot_eff)}
  r_walk_noise={st(f.posterior.r_walk_noise, dec=3)} mean(r_w_n^2)={np.mean(np.array(f.posterior.r_walk_noise)**2):.6f}"""
    )
    if "seasonality_beta1" in f.posterior:
        b1 = f.posterior.seasonality_beta1
        print(
            f"  seasonality_beta1={st(b1)}, R0(Jan 1) / R0(July 1) = {st((1 + b1) / (1-b1))}\n"
            f'  equivalent NPI effect of "summer July 1" (vs "Jan 1") = {st(100*(1 - (1 - b1) / (1 + b1)))}'
        )
        if args.write_seasonality:
            with open(args.write_seasonality, "wt") as f:
                f.write(",".join(str(b) for b in np.array(b1)) + "\n")
    efs = [st(100 * (1 - np.exp(-d)), dec=2, short=True) for d in f.posterior.alpha_i.T]
    print("\n  effects(95% CI):")
    while efs:
        print("    ", ", ".join(efs[:5]))
        efs = efs[5:]
    print()
    if args.plot_dists:
        import seaborn as sns
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Dists for {s['exp_tag']}\n{s['exp_config']}")

        axes[0, 0].set_title(
            f"r_walk_noise: {st(f.posterior.r_walk_noise, ci=False)}\nmean(x^2)={np.mean(np.array(f.posterior.r_walk_noise)**2):.4f}"
        )
        axes[0, 0].set_xlim([-0.6, 0.6])
        sns.histplot(
            np.array(f.posterior.r_walk_noise).flatten(), ax=axes[0, 0], bins=50
        )

        axes[1, 0].set_title(f"basic_R: {st(f.posterior.basic_R, ci=False)}")
        axes[1, 0].set_xlim([0.5, 3.0])
        sns.histplot(np.array(f.posterior.basic_R).flatten(), ax=axes[1, 0], bins=50)

        axes[0, 1].set_title(f"total NPI effect: {st(tot_eff, short=True)}")
        axes[0, 1].set_xlim([60, 80])
        sns.histplot(np.array(tot_eff).flatten(), ax=axes[0, 1], bins=50)

        axes[1, 1].set_xlim([-0.25, 0.75])
        axes[1, 1].set_title(f"seasonality beta_1: NA")
        if "seasonality_beta1" in f.posterior:
            axes[1, 1].set_title(f"seasonality beta_1: {st(b1, short=True)}")
            sns.histplot(np.array(b1).flatten(), ax=axes[1, 1], bins=50)

        plt.tight_layout()
        plt.savefig(args.summary_json.replace("_summary.json", "_dists.pdf"))
