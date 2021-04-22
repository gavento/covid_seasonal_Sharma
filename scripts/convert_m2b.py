"""
A script converting Sharma et al. input data (non-featurized)
to Brauner et al. input data (different columns, featurized).
"""

import os
import sys

sys.path.append(os.getcwd())

import argparse

import numpy as np
import pandas as pd
from epimodel import preprocess_data

# FROM: Area,Date,Country,New Cases,New Deaths,[NPIs,]
#   Public Outdoor Gathering Person Limit,Public Indoor Gathering Person Limit,Private Outdoor Gathering Person Limit,Private Indoor Gathering Person Limit,Public Outdoor Household Limit,Public Indoor Household Limit,Private Outdoor Household Limit,Private Indoor Household Limit,Mandatory Mask Wearing,Some Face-to-Face Businesses Closed,Gastronomy Closed,Leisure Venues Closed,Retail Closed,All Face-to-Face Businesses Closed,Stay at Home Order,Curfew,Childcare Closed,Primary Schools Closed,Secondary Schools Closed,Universities Away
# TO: Country Code,Date,Region Name,Confirmed,Active,Deaths,[NPIs,]
#   Some Face-to-Face Businesses Closed,Gastronomy Closed,Leisure Venues Closed,Retail Closed,Curfew,Childcare Closed,Primary Schools Closed,Secondary Schools Closed,Universities Away,Public Indoor Gathering Person Limit - 1,Public Indoor Gathering Person Limit - 2,Public Indoor Gathering Person Limit - 10,Public Indoor Gathering Person Limit - 30,Extra Public Indoor Household Limit,Private Indoor Gathering Person Limit - 1,Private Indoor Gathering Person Limit - 2,Private Indoor Gathering Person Limit - 10,Private Indoor Gathering Person Limit - 30,Extra Private Indoor Household Limit,Mandatory Mask Wearing >= 3
#   (WAS: Mask Wearing,Symptomatic Testing,Gatherings <1000,Gatherings <100,Gatherings <10,Some Businesses Suspended,Most Businesses Suspended,School Closure,University Closure,Stay Home Order,Travel Screen/Quarantine,Travel Bans,Public Transport Limited,Internal Movement Limited,Public Information Campaigns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("from_csv")
    parser.add_argument("to_csv")
    parser.add_argument("--new_variant", default="")
    args = parser.parse_args()

    data = preprocess_data(args.from_csv, skipcases=0, skipdeaths=0)
    data.featurize()
    if args.new_variant:
        data.mask_new_variant(
            new_variant_fraction_fname=args.new_variant,
        )
    data.mask_from_date("2021-01-09")

    #    cs = d.columns
    #    assert cs[4] == "New Deaths"
    #    npis = cs[5:]
    #    print(f"Input: {len(npis)} NPIs: {npis!r}")

    #    names = d.apply(lambda r: f"{r['Country']}/{r['Area']}", axis=1)
    nrs = data.nDs * data.nRs

    def nf(x):
        return np.nan if x is np.ma.masked else np.float32(x)
    rs = []
    for Ri, R in enumerate(data.Rs):
        for Di, D in enumerate(data.Ds):
            rs.append(
                {
                    "Country Code": R,
                    "Date": D,
                    "Region Name": f"{data.Cs[Ri]}/{R}",
                    "Confirmed": nf(data.new_cases[Ri, Di]),
                    "Active": np.nan,
                    "Deaths": nf(data.new_deaths[Ri, Di]),
                }
            )
            for CMi, CM in enumerate(data.CMs):
                rs[-1][CM] = data.active_cms[Ri, CMi, Di]
    df = pd.DataFrame(rs)
    df.to_csv(args.to_csv, index=False)


if __name__ == "__main__":
    main()
