import numpy as np
import arviz
import json
import argparse


def st(d, ci=0.95, dec=3, short=False):
    d = np.array(d)
    c = np.quantile(d, [0.5 - ci / 2, 0.5, 0.5 + ci / 2])
    if short:
        return f"{np.mean(d):.{dec}f} ({c[0]:.{dec}f}..{c[2]:.{dec}f})"
    return f"{np.mean(d):.{dec}f} (sd={np.std(d):.{dec}f}, 95%CI {c[0]:.{dec}f} .. {c[2]:.{dec}f})"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_json")
    parser.add_argument("-w", "--write_seasonality", type=str)
    args = parser.parse_args()

    with open(args.summary_json) as f:
        s = json.load(f)
    f = arviz.from_netcdf(args.summary_json.replace("_summary.json", "_full.netcdf"))
    chains = len(f.posterior.chain)
    print(
        f"""Loaded {args.summary_json}

  experiment: {s['exp_tag']} {s['exp_config']}, chains: {chains}x {len(s['warmup']['num_steps']) // chains} + {len(s['sample']['num_steps']) // chains} samples
  model:      {s['model_name']} {s['model_kwargs']}, config={s['model_config_name']!r}

  rhat={s['rhat']['lower']:.3f}-{s['rhat']['upper']:.3f}, divergences={s['divergences']}, accept_prob={st(s['sample']['mean_accept_prob'])}, total_runtime={s['total_runtime']:.2f} s
  basic_R={st(f.posterior.basic_R)}, total effect={st(100*(1-np.exp(-np.sum(f.posterior.alpha_i, axis=-1))))}
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
