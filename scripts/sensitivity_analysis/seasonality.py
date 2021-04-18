import os
import sys

sys.path.append(os.getcwd())  # add current working directory to the path

import argparse
import json
import subprocess
from datetime import datetime

import numpyro
from epimodel import EpidemiologicalParameters, preprocess_data, run_model
from epimodel.models.models import seasonality_model
from epimodel.script_utils import *

argparser = argparse.ArgumentParser()

add_argparse_arguments(argparser)
argparser.add_argument(
    "--output_base",
    dest="output_base",
    type=str,
    default="",
)

argparser.add_argument(
    "--r_walk_noise_scale_prior",
    dest="r_walk_noise_scale_prior",
    type=float,
    help="r_walk_noise_scale_prior",
    default=0.15,
)
argparser.add_argument(
    "--basic_R_mean",
    dest="basic_R_mean",
    type=float,
    default=1.35,
    help="Basic R mean",
)
argparser.add_argument(
    "--basic_R_scale",
    dest="basic_R_scale",
    type=float,
    default=0.3,
    help="Basic R scale",
)
argparser.add_argument(
    "--seasonality_max_R_day",
    dest="seasonality_max_R_day",
    type=int,
    default=1,
    help="Day of the seasonally-highest R (1..365)",
)
args = argparser.parse_args()

if __name__ == "__main__":
    numpyro.set_host_device_count(args.num_chains)

    if not args.output_base:
        base_outpath = generate_base_output_dir(
            args.model_type, args.model_config, args.exp_tag
        )
        ts_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_base = os.path.join(base_outpath, f"{ts_str}-{os.getpid()}")
    log_output = f"{args.output_base}.log"
    summary_output = f"{args.output_base}_summary.json"
    full_output = f"{args.output_base}_full.netcdf"
    logprocess = subprocess.Popen(["/bin/tee", log_output], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    os.close(sys.stdout.fileno())
    os.dup2(logprocess.stdin.fileno(), sys.stdout.fileno())
    os.close(sys.stderr.fileno())
    os.dup2(logprocess.stdin.fileno(), sys.stderr.fileno())

    print(f"Running Sensitivity Analysis {__file__} with config:")
    config = load_model_config(args.model_config)
    pprint_mb_dict(config)

    print("Loading Data")
    data = preprocess_data(get_data_path())
    data.featurize(**config["featurize_kwargs"])
    data.mask_new_variant(
        new_variant_fraction_fname=get_new_variant_path(),
    )
    data.mask_from_date("2021-01-09")

    print("Loading EpiParam")
    ep = EpidemiologicalParameters()

    model_func = seasonality_model
    ta = get_target_accept_from_model_str(args.model_type)
    td = get_tree_depth_from_model_str(args.model_type)

    basic_R_prior = {
        "mean": args.basic_R_mean,
        "type": "trunc_normal",
        "variability": args.basic_R_scale,
    }

    model_build_dict = config["model_kwargs"]
    model_build_dict["r_walk_noise_scale_prior"] = args.r_walk_noise_scale_prior
    model_build_dict["basic_R_prior"] = basic_R_prior
    model_build_dict["seasonality_max_R_day"] = args.seasonality_max_R_day

    posterior_samples, _, info_dict, _ = run_model(
        model_func,
        data,
        ep,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        num_warmup=args.num_warmup,
        target_accept=ta,
        max_tree_depth=td,
        model_kwargs=model_build_dict,
        save_results=True,
        output_fname=full_output,
        chain_method="parallel",
    )

    info_dict["model_config_name"] = args.model_config
    info_dict["model_kwargs"] = config["model_kwargs"]
    info_dict["featurize_kwargs"] = config["featurize_kwargs"]
    info_dict["start_dt"] = ts_str
    info_dict["exp_tag"] = args.exp_tag
    info_dict["exp_config"] = {
        "r_walk_noise_scale_prior": args.r_walk_noise_scale_prior,
        "basic_R_prior": basic_R_prior,
        "seasonality_max_R_day": args.seasonality_max_R_day,
    }
    info_dict["cm_names"] = data.CMs
    info_dict["data_path"] = get_data_path()

    # also need to add sensitivity analysis experiment options to the summary dict!
    summary = load_keys_from_samples(
        get_summary_save_keys(), posterior_samples, info_dict
    )
    with open(summary_output, "w") as f:
        json.dump(info_dict, f, ensure_ascii=False, indent=4)
