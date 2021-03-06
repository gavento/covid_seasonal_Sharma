import os
import subprocess
import sys
from pathlib import Path

sys.path.append(os.getcwd())  # add current working directory to the path

import argparse
import json
from datetime import datetime

from epimodel import EpidemiologicalParameters, preprocess_data, run_model
from epimodel.script_utils import *

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--output_base",
    dest="output_base",
    type=str,
    help="Override destination path prefix (adding '.log', '_summary.json', '_full.netcdf')",
    default="",
)

argparser.add_argument(
    "--basic_R_mean",
    dest="basic_R_mean",
    type=float,
    help="Basic R mean",
)

argparser.add_argument(
    "--basic_R_scale",
    dest="basic_R_scale",
    type=float,
    help="Basic R scale",
)

argparser.add_argument(
    "--brauner_params",
    action="store_true",
    help="Use the epidemiological parameters from Brauner et al. model",
)

add_argparse_arguments(argparser)
args = argparser.parse_args()

import numpyro

numpyro.set_host_device_count(args.num_chains)

if __name__ == "__main__":

    if not args.output_base:
        base_outpath = generate_base_output_dir(
            args.model_type, args.model_config, args.exp_tag
        )
        ts_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_base = os.path.join(base_outpath, f"{ts_str}-{os.getpid()}")
    Path(args.output_base).parent.mkdir(parents=True, exist_ok=True)
    log_output = f"{args.output_base}.log"
    summary_output = f"{args.output_base}_summary.json"
    full_output = f"{args.output_base}_full.netcdf"
    logprocess = subprocess.Popen(["/usr/bin/tee", log_output], stdin=subprocess.PIPE)
    os.close(sys.stdout.fileno())
    os.dup2(logprocess.stdin.fileno(), sys.stdout.fileno())
    os.close(sys.stderr.fileno())
    os.dup2(logprocess.stdin.fileno(), sys.stderr.fileno())


    print(f"Running Sensitivity Analysis {__file__} with config:")
    config = load_model_config(args.model_config)
    pprint_mb_dict(config)

    print("Loading Data")
    data = load_preprecess_data(config)

    print("Loading EpiParam")
    if args.brauner_params:
        ep = EpidemiologicalParameters(
            generation_interval={"mean": 5.06, "sd": 2.11, "dist": "gamma"},
            incubation_period={"mean": 1.53, "sd": 0.418, "dist": "gamma"},
        )
        # Generate directly from infection dists
        ep.DPC = ep.generate_dist_vector({"mean": 10.9, "disp": 5.4, "dist": "negbiom"}, int(1e7), ep.cd_truncation)
        ep.DPD = ep.generate_dist_vector({"mean": 21.8, "disp": 14.2, "dist": "negbiom"}, int(1e7), ep.dd_truncation)
        # Make sure these are not used further
        ep.onset_to_case_delay = None
        ep.onset_to_death_delay = None
    else:
        ep = EpidemiologicalParameters()

    model_func = get_model_func_from_str(args.model_type)
    ta = get_target_accept_from_model_str(args.model_type)
    td = get_tree_depth_from_model_str(args.model_type)

    basic_R_prior = {
        "mean": args.basic_R_mean,
        "type": "trunc_normal",
        "variability": args.basic_R_scale,
    }

    model_build_dict = config["model_kwargs"]
    model_build_dict["basic_R_prior"] = basic_R_prior

    print("model_build_dict:", model_build_dict)

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
    info_dict["exp_config"] = {"basic_R_prior": basic_R_prior}
    info_dict["cm_names"] = data.CMs
    info_dict["data_path"] = get_data_path()

    # also need to add sensitivity analysis experiment options to the summary dict!
    summary = load_keys_from_samples(
        get_summary_save_keys(), posterior_samples, info_dict
    )
    with open(summary_output, "w") as f:
        json.dump(info_dict, f, ensure_ascii=False, indent=4)
