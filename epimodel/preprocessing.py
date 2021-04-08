"""
:code:`preprocessed_data.py`

PreprocessedData Class definition, plus some utility functions.
"""
import numpy as np
import pandas as pd


def preprocess_data(
    data_path,
    start_date="2020-08-01",
    end_date="2021-01-22",
    npi_start_col=3,
    skipcases=8,
    skipdeaths=26,
    household_feature_processing="implicit",
):
    """
    Process data, return PreprocessedData() object

    :param data_path: merged data path
    :param start_date: window of analysis start date
    :param end_date: window of analysis end date
    :param npi_start_col: start column of npis
    :param skipcases: Number of days of cases to mask at the start of the period. We mask since these
                      cases were generated by infections we cannot account for (they were outside of this period).
    :param skipdeaths: Number of days of deaths to mask at the start of the period. We mask since these
                      cases were generated by infections we cannot account for (they were outside of this period).
    :param household_feature_processing: either implicit or raw. implicit takes the implied gathering limit from the
                                         the household limit, while raw ignore this
    :return: PreprocessedData object containing loaded data, but data that has not yet been featurised.
    """

    def set_household_limits(active_cms, household_NPI_index, gathering_NPI_index):
        """
        This function implements implicit processing of household NPIs. i.e., if the
        gathering limit is X, then the household limit is also set to X if no household
        limit has been recorded. In other words, if the gathering limit is stricter than
        the household limit (e.g., 6 people from 10 households), the household limit is set
        to the gathering limit, since worst case is that every person comes from their own
        household.

        :param active_CMs: nRs x nCMs x nDs activation matrix
        :param household_NPI_index: household index
        :param gathering_NPI_index: respective gathering index
        :return: new active_cms matrix
        """
        nRs, _, nDs = active_cms.shape
        new_acms = np.copy(active_cms)
        for r in range(nRs):
            for day in range(nDs):
                if (
                    active_cms[r, household_NPI_index, day] == 0
                    or active_cms[r, gathering_NPI_index, day]
                    < active_cms[r, household_NPI_index, day]
                ):
                    new_acms[r, household_NPI_index, day] = active_cms[
                        r, gathering_NPI_index, day
                    ]
        return new_acms

    df = pd.read_csv(
        data_path, parse_dates=["Date"], infer_datetime_format=True
    ).set_index(["Area", "Date"])

    Ds = list(df.index.levels[1])

    if end_date is not None:
        last_ts = pd.to_datetime(end_date)
        # +1 because we want to include the last day
        Ds = Ds[: (1 + Ds.index(last_ts))]

    if start_date is not None:
        start_ts = pd.to_datetime(start_date)
        Ds = Ds[Ds.index(start_ts) :]

    print(f"Processing data from {Ds[0]} to {Ds[-1]}")

    Rs = list(df.index.levels[0])
    CMs = list(df.columns[npi_start_col:])

    nRs = len(Rs)
    nDs = len(Ds)
    nCMs = len(CMs)

    Cs = []

    for r in Rs:
        c = df.loc[r]["Country"][0]
        Cs.append(c)

    # sort the countries by name. we need to preserve the ordering
    unique_Cs = sorted(list(set(Cs)))
    nCs = len(unique_Cs)

    C_indices = []
    for uc in unique_Cs:
        a_indices = np.nonzero([uc == c for c in Cs])[0]
        C_indices.append(a_indices)

    active_cms = np.zeros((nRs, nCMs, nDs))
    new_cases = np.ma.zeros((nRs, nDs), dtype="int")
    new_deaths = np.ma.zeros((nRs, nDs), dtype="int")

    for r_i, r in enumerate(Rs):
        r_df = df.loc[r].loc[Ds]
        new_cases.data[r_i, :] = r_df["New Cases"]
        new_deaths.data[r_i, :] = r_df["New Deaths"]

        for cm_i, cm in enumerate(CMs):
            active_cms[r_i, cm_i, :] = r_df[cm]

    if household_feature_processing == "implicit":
        pairs = [
            ("Public Outdoor Gathering Person Limit", "Public Outdoor Household Limit"),
            ("Public Indoor Gathering Person Limit", "Public Indoor Household Limit"),
            (
                "Private Outdoor Gathering Person Limit",
                "Private Outdoor Household Limit",
            ),
            ("Private Indoor Gathering Person Limit", "Private Indoor Household Limit"),
        ]

        for pair in pairs:
            gathering_ind = CMs.index(pair[0])
            household_ind = CMs.index(pair[1])
            active_cms = set_household_limits(active_cms, household_ind, gathering_ind)
    elif household_feature_processing == "raw":
        pass
    else:
        raise ValueError("household_feature_processing must be in [implicit, raw]")

    # mask days where there are negative cases or deaths - because this
    # is clearly wrong
    new_cases[new_cases < 0] = np.ma.masked
    new_deaths[new_deaths < 0] = np.ma.masked
    # do this to make sure.
    new_cases.data[new_cases.data < 0] = 0
    new_deaths.data[new_deaths.data < 0] = 0

    new_cases[:, :skipcases] = np.ma.masked
    new_deaths[:, :skipdeaths] = np.ma.masked

    return PreprocessedData(
        Rs, Ds, CMs, new_cases, new_deaths, active_cms, Cs, unique_Cs, C_indices
    )


class PreprocessedData(object):
    """
    PreprocessedData Class

    Class to hold data which is subsequently passed onto a numpyro model. Mostly a data wrapper, with some utility
    functions.
    """

    def __init__(
        self, Rs, Ds, CMs, new_cases, new_deaths, active_cms, Cs, unique_Cs, C_indices
    ):
        """

        :param Rs: list of regions
        :param Ds: list of days
        :param CMs: list of intervention names
        :param new_cases: numpy.ma.array of daily cases
        :param new_deaths: numpy.ma.array of daily deaths
        :param active_cms:
        :param Cs: list of countries (corresponding to the list of regions)
        :param unique_Cs:
        :param C_indices:
        """
        super().__init__()
        self.Rs = Rs
        self.Ds = Ds
        self.CMs = CMs
        self.new_cases = new_cases
        self.new_deaths = new_deaths
        self.active_cms = active_cms
        self.Cs = Cs
        self.unique_Cs = unique_Cs
        self.C_indices = C_indices

        # the RC mat is used for partial pooling to grab the right country specific noise
        # for that particular region
        self.RC_mat = np.zeros((self.nRs, self.nCs))
        for r_i, c in enumerate(self.Cs):
            C_ind = self.unique_Cs.index(c)
            self.RC_mat[r_i, C_ind] = 1

        # 1-based day of the year. "January 1" is 1
        self.Ds_day_of_year = np.array([t.dayofyear for t in self.Ds], dtype=np.int32)

        self.featurized = False

    @property
    def nCMs(self):
        return len(self.CMs)

    @property
    def nRs(self):
        return len(self.Rs)

    @property
    def nCs(self):
        return len(self.unique_Cs)

    @property
    def nDs(self):
        return len(self.Ds)

    def drop_npi_by_index(self, npi_index):
        include_npi = np.ones(self.nCMs, dtype=np.bool)
        include_npi[npi_index] = False

        cm_names = np.array(self.CMs)[include_npi].tolist()
        self.active_cms = self.active_cms[:, include_npi, :]
        self.CMs = cm_names

    def featurize(
        self,
        drop_npi_filter=None,
        public_gathering_thresholds=None,
        private_gathering_thresholds=None,
        mask_thresholds=None,
        household_stays_on=True,
        household_upper_limit=7,
        gatherings_aggregation="drop_outdoor",
        gatherings_aggregation_type="weaker",
        keep_merged_value=False,
    ):
        """
        Featurise i.e., convert raw NPIs to binary features

        :param drop_npi_filter: list of dictionaries containing npis to ignore
        :param public_gathering_thresholds: public gathering thresholds
        :param private_gathering_thresholds: private gathering thresholds
        :param mask_thresholds: mask thresholds
        :param household_stays_on: whether the household feature should stay on when we hit a gathering limit of 2 or less
        :param household_upper_limit: gathering limit under which to count the household restriction
        :param gatherings_aggregation: how to aggregate gatherings. either `drop_outdoor, out_in, pub_priv`
        :param gatherings_aggregation_type: either `weaker` or `stricter`
        :param keep_merged_value: keep merged values (gatherings)
        """
        if self.featurized is True:
            print(
                "Data has already been featurized. New featurisation will not take place"
            )
            return

        binary_npis = [
            "Some Face-to-Face Businesses Closed",
            "Gastronomy Closed",
            "Leisure Venues Closed",
            "Retail Closed",
            "Curfew",
            "Primary Schools Closed",
            "Secondary Schools Closed",
            "Universities Away",
        ]

        def aggregate_numerical_npis(agg_type, cm_a_ind, cm_b_ind, active_cms_in):
            active_cms = np.copy(active_cms_in)
            if agg_type == "stricter":
                cm_a_vals = active_cms[:, cm_a_ind, :]
                cm_a_vals[cm_a_vals == 0] = np.inf
                cm_b_vals = active_cms[:, cm_b_ind, :]
                cm_b_vals[cm_b_vals == 0] = np.inf

                agg_vals = np.minimum(cm_a_vals, cm_b_vals)
                agg_vals[agg_vals == np.inf] = 0

            elif agg_type == "weaker":
                cm_a_vals = active_cms[:, cm_a_ind, :]
                cm_a_vals[cm_a_vals == 0] = np.inf
                cm_b_vals = active_cms[:, cm_b_ind, :]
                cm_b_vals[cm_b_vals == 0] = np.inf

                agg_vals = np.maximum(cm_a_vals, cm_b_vals)
                agg_vals[agg_vals == np.inf] = 0

            return agg_vals

        if drop_npi_filter is None:
            drop_npi_filter = [
                {
                    "query": "All Face-to-Face Businesses Closed",
                    "type": "equals",
                },  # same as the retail feature
            ]

        if public_gathering_thresholds is None:
            public_gathering_thresholds = [1, 2, 10, 30]

        if private_gathering_thresholds is None:
            private_gathering_thresholds = [1, 2, 10, 30]

        if mask_thresholds is None:
            mask_thresholds = [3]

        if gatherings_aggregation is None or gatherings_aggregation == "none":
            agg = None

            gathering_household_npi_pairs = [
                (
                    "Public Outdoor Gathering Person Limit",
                    "Public Outdoor Household Limit",
                ),
                (
                    "Public Indoor Gathering Person Limit",
                    "Public Indoor Household Limit",
                ),
                (
                    "Private Outdoor Gathering Person Limit",
                    "Private Outdoor Household Limit",
                ),
                (
                    "Private Indoor Gathering Person Limit",
                    "Private Indoor Household Limit",
                ),
            ]

        elif gatherings_aggregation == "pub_priv":
            agg = [
                (
                    "Public Outdoor Gathering Person Limit",
                    "Private Outdoor Gathering Person Limit",
                    "Outdoor Gathering Person Limit",
                ),
                (
                    "Public Indoor Gathering Person Limit",
                    "Private Indoor Gathering Person Limit",
                    "Indoor Gathering Person Limit",
                ),
                (
                    "Public Outdoor Household Limit",
                    "Private Outdoor Household Limit",
                    "Outdoor Household Limit",
                ),
                (
                    "Public Indoor Household Limit",
                    "Private Indoor Household Limit",
                    "Indoor Household Limit",
                ),
            ]

            new_active_cms = np.copy(self.active_cms)
            for cm_a, cm_b, new_name in agg:
                # i need to properly deal with the zeros now too
                cm_a_ind = self.CMs.index(cm_a)
                cm_b_ind = self.CMs.index(cm_b)

                agg_vals = aggregate_numerical_npis(
                    gatherings_aggregation_type, cm_a_ind, cm_b_ind, self.active_cms
                )

                new_active_cms[:, cm_a_ind, :] = agg_vals
                self.CMs[cm_a_ind] = new_name

            gathering_household_npi_pairs = [
                ("Outdoor Gathering Person Limit", "Outdoor Household Limit"),
                ("Indoor Gathering Person Limit", "Indoor Household Limit"),
            ]
            self.active_cms = new_active_cms

        elif gatherings_aggregation == "out_in":
            agg = [
                (
                    "Public Outdoor Gathering Person Limit",
                    "Public Indoor Gathering Person Limit",
                    "Public Gathering Person Limit",
                ),
                (
                    "Private Indoor Gathering Person Limit",
                    "Private Outdoor Gathering Person Limit",
                    "Private Gathering Person Limit",
                ),
                (
                    "Public Outdoor Household Limit",
                    "Public Indoor Household Limit",
                    "Public Household Limit",
                ),
                (
                    "Private Outdoor Household Limit",
                    "Private Indoor Household Limit",
                    "Private Household Limit",
                ),
            ]

            new_active_cms = np.copy(self.active_cms)
            for cm_a, cm_b, new_name in agg:
                # i need to properly deal with the zeros now too
                cm_a_ind = self.CMs.index(cm_a)
                cm_b_ind = self.CMs.index(cm_b)

                agg_vals = aggregate_numerical_npis(
                    gatherings_aggregation_type, cm_a_ind, cm_b_ind, self.active_cms
                )

                new_active_cms[:, cm_a_ind, :] = agg_vals
                self.CMs[cm_a_ind] = new_name

            gathering_household_npi_pairs = [
                ("Public Gathering Person Limit", "Public Household Limit"),
                ("Private Gathering Person Limit", "Private Household Limit"),
            ]
            self.active_cms = new_active_cms

        elif gatherings_aggregation == "drop_outdoor":
            drop_npi_filter.append({"query": "Outdoor", "type": "includes"})

            gathering_household_npi_pairs = [
                (
                    "Public Indoor Gathering Person Limit",
                    "Public Indoor Household Limit",
                ),
                (
                    "Private Indoor Gathering Person Limit",
                    "Private Indoor Household Limit",
                ),
            ]

            print(
                "Note: under drop_outdoor gathering aggregation, the gatherings_aggregation_type is disregarded"
            )
        else:
            raise ValueError(
                "gatherings_aggregation must be in [drop_outdoor, out_in, pub_priv, none]"
            )

        mask_npi = "Mandatory Mask Wearing"

        nRs, _, nDs = self.active_cms.shape

        new_active_cms = np.zeros((nRs, 0, nDs))

        cm_names = []
        for bin_npi in binary_npis:
            old_index = self.CMs.index(bin_npi)
            new_cm_feature = self.active_cms[:, old_index, :]
            new_active_cms = np.append(
                new_active_cms, new_cm_feature.reshape((nRs, 1, nDs)), axis=1
            )
            cm_names.append(bin_npi)

        for gath_npi, hshold_npi in gathering_household_npi_pairs:
            gath_npi_ind = self.CMs.index(gath_npi)
            hshold_npi_ind = self.CMs.index(hshold_npi)

            if "Private" in gath_npi:
                thresholds = private_gathering_thresholds
            else:
                thresholds = public_gathering_thresholds

            if keep_merged_value:
                new_cm_feature = self.active_cms[:, gath_npi_ind, :]
                new_active_cms = np.append(
                    new_active_cms, new_cm_feature.reshape((nRs, 1, nDs)), axis=1
                )
                cm_names.append(f"{gath_npi}")

            for t in thresholds:
                new_cm_feature = np.logical_and(
                    self.active_cms[:, gath_npi_ind, :] > 0,
                    self.active_cms[:, gath_npi_ind, :] < t + 1,
                )
                new_active_cms = np.append(
                    new_active_cms, new_cm_feature.reshape((nRs, 1, nDs)), axis=1
                )
                cm_names.append(f"{gath_npi} - {t}")

            if household_stays_on:
                # if there is a gathering ban under the upper limit people, and the household limit is 2
                is_relevant_gath_ban = np.logical_and(
                    self.active_cms[:, gath_npi_ind, :] < household_upper_limit,
                    self.active_cms[:, gath_npi_ind, :] > 0,
                )
                household_feature = np.logical_and(
                    is_relevant_gath_ban, self.active_cms[:, hshold_npi_ind, :] < 3
                )
                new_active_cms = np.append(
                    new_active_cms, household_feature.reshape((nRs, 1, nDs)), axis=1
                )
                cm_names.append(f"Extra {hshold_npi}")
            else:
                # i.e., is there a ban between 2 and the upper limit (default between 3 and 10 inclusive)
                # and an additional household limit of 2 people on that.
                is_relevant_gath_ban = np.logical_and(
                    self.active_cms[:, gath_npi_ind, :] > 2,
                    self.active_cms[:, gath_npi_ind, :] < household_upper_limit,
                )
                household_feature = np.logical_and(
                    is_relevant_gath_ban, self.active_cms[:, hshold_npi_ind, :] < 3
                )
                new_active_cms = np.append(
                    new_active_cms, household_feature.reshape((nRs, 1, nDs)), axis=1
                )
                cm_names.append(f"Extra {hshold_npi}")

        mask_ind = self.CMs.index(mask_npi)
        for t in mask_thresholds:
            mask_feature = self.active_cms[:, mask_ind, :] > t - 1
            new_active_cms = np.append(
                new_active_cms, mask_feature.reshape((nRs, 1, nDs)), axis=1
            )
            cm_names.append(f"{mask_npi} >= {t}")

        nCMs = len(cm_names)
        include_npi = np.ones(nCMs, dtype=np.bool)
        for cm_i, name in enumerate(cm_names):
            for filter_dict in drop_npi_filter:
                if filter_dict["type"] == "equals":
                    if name == filter_dict["query"]:
                        include_npi[cm_i] = False

                if filter_dict["type"] == "includes":
                    if filter_dict["query"] in name:
                        include_npi[cm_i] = False

        cm_names = np.array(cm_names)[include_npi].tolist()
        self.active_cms = new_active_cms[:, include_npi, :]
        self.CMs = cm_names
        self.featurized = True

    def mask_region_by_index(self, region_index, days_shown=90):
        """
        mask end of region

        :param region_index: region index
        :param days_shown: num days to show
        """
        self.new_cases[region_index, days_shown:] = np.ma.masked
        self.new_deaths[region_index, days_shown:] = np.ma.masked

    def remove_region_by_index(self, r_i):
        """
        fully remove region
        :param r_i: region index
        """
        # remove the index
        del self.Rs[r_i]
        del self.Cs[r_i]
        self.new_cases = np.delete(self.new_cases, r_i, axis=0)
        self.new_deaths = np.delete(self.new_deaths, r_i, axis=0)
        self.active_cms = np.delete(self.active_cms, r_i, axis=0)

        self.unique_Cs = sorted(list(set(self.Cs)))
        C_indices = []
        for uc in self.unique_Cs:
            a_indices = np.nonzero([uc == c for c in self.Cs])[0]
            C_indices.append(a_indices)

        self.C_indices = C_indices

        self.RC_mat = np.zeros((self.nRs, self.nCs))
        for r_i, c in enumerate(self.Cs):
            C_ind = self.unique_Cs.index(c)
            self.RC_mat[r_i, C_ind] = 1

    def mask_new_variant(
        self,
        maximum_fraction_voc=0.1,
        new_variant_fraction_fname="../data/nuts3_new_variant_fraction.csv",
        extra_days_cases=0, #prev 5
        extra_days_deaths=6, #prev 11
    ):
        """
        mask new variant

        :param maximum_fraction_voc: maximum fraction
        :param new_variant_fraction_fname: filename with raw data
        :param extra_days_cases: extra days of cases
        :param extra_days_deaths: extra days of deaths
        :return:
        """
        variant_df = pd.read_csv(new_variant_fraction_fname)
        variant_df["date"] = pd.to_datetime(variant_df["date"], format="%Y-%m-%d")
        variant_df["nuts3"] = variant_df["nuts3"].replace(
            ["Buckinghamshire"], "Buckinghamshire CC"
        )

        variant_df = variant_df[variant_df["frac"] > maximum_fraction_voc]
        regions_to_mask = np.unique(variant_df["nuts3"])
        variant_df = variant_df.set_index(["nuts3"])
        mask_forward_dates = []
        for region in regions_to_mask:
            try:
                mask_forward_dates.append(variant_df.loc[region]["date"][0])
            except:
                mask_forward_dates.append(variant_df.loc[region]["date"])

        for i in range(len(mask_forward_dates)):
            if mask_forward_dates[i] in self.Ds:
                self.new_cases[
                    self.Rs.index(regions_to_mask[i]),
                    self.Ds.index(mask_forward_dates[i]) + extra_days_cases :,
                ] = np.ma.masked
                self.new_deaths[
                    self.Rs.index(regions_to_mask[i]),
                    self.Ds.index(mask_forward_dates[i]) + extra_days_deaths :,
                ] = np.ma.masked

    def mask_reopening(self, option, npis_to_exclude=None):
        if npis_to_exclude is None:
            npis_to_exclude = [
                "Primary Schools Closed",
                "Secondary Schools Closed",
                "Universities Away",
            ]
        npis_to_include = [CM for CM in self.CMs if CM not in npis_to_exclude]

        active_CMs = self.active_cms
        active_CMs = (active_CMs > 0).astype(int)
        changes = []
        for i in range(active_CMs.shape[0]):
            change = np.zeros((len(self.CMs), len(self.Ds)))
            change[:, 1:] = active_CMs[i, :, 1:] - active_CMs[i, :, :-1]
            changes.append(change)
        self.number_masked = []
        for r in range(active_CMs.shape[0]):
            days_masked_npi = []
            for npi in npis_to_include:
                npi_index = self.CMs.index(npi)
                changes_region = changes[r][npi_index]
                starts = list(np.where(changes_region == -1)[0])
                ends = list(np.where(changes_region == 1)[0])
                # print(starts, ends)
                if len(starts) > 0:
                    if len(ends) > 0:
                        if ends[-1] < starts[-1]:
                            ends.append(len(self.Ds) - 1)
                        if ends[0] < starts[0]:
                            ends = ends[1:]
                    else:
                        ends.append(len(self.Ds) - 1)
                    # print(starts, ends)
                    for i in range(len(starts)):
                        if option == 3:
                            days_masked_npi.append(ends[i] - starts[i])
                            self.new_cases[r, starts[i] : ends[i]] = np.ma.masked
                            self.new_deaths[r, starts[i] : ends[i]] = np.ma.masked
                        if option == 4:
                            days_masked_npi.append(len(self.Ds) - 1 - starts[i])
                            self.new_cases[r, starts[i] :] = np.ma.masked
                            self.new_deaths[r, starts[i] :] = np.ma.masked
                else:
                    days_masked_npi.append(0)
            self.number_masked.append(max(days_masked_npi))

    def mask_from_date(self, date_str, extra_days_cases=5, extra_days_deaths=11):
        """
        mask from a certain date

        :param date_str:
        :param extra_days_cases:
        :param extra_days_deaths:
        """
        dt = pd.to_datetime(date_str)
        if dt not in self.Ds:
            raise ValueError("Requested date_str not in Ds")

        date_index = self.Ds.index(dt)
        self.new_cases[:, date_index + extra_days_cases :] = np.ma.masked
        self.new_deaths[:, date_index + extra_days_deaths :] = np.ma.masked
