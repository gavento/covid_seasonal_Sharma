import numpy as np
import arviz
import json
import argparse


def st(d, ci=0.95, dec=2, short=False):
    d = np.array(d)
    c = np.quantile(d, [0.5 - ci / 2, 0.5, 0.5 + ci / 2])
    if short:
        return f"{np.mean(d):.{dec}f} ({c[0]:.{dec}f}..{c[2]:.{dec}f})"    
    return f"{np.mean(d):.{dec}f} (med {c[1]:.{dec}f}, 95%CI {c[0]:.{dec}f} .. {c[2]:.{dec}f})"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_json")
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
  basic_R={st(f.posterior.basic_R)}, r_walk_noise={st(f.posterior.r_walk_noise)}, total effect={st(100*(1-np.exp(-np.sum(f.posterior.alpha_i, axis=-1))))}""")
    if "seasonality_beta1" in f.posterior:
        print(f"  seasonality_beta1={st(f.posterior.seasonality_beta1)}")
    efs = [st(100*(1-np.exp(-d)), short=True) for d in f.posterior.alpha_i.T]
    print("\n  effects(95% CI):")
    while efs:
        print("    ", ', '.join(efs[:5]))
        efs = efs[5:]
    print()
