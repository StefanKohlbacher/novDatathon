import pandas as pd
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

df_rte = pd.read_csv(os.path.join(current_dir, "../data/data_files/rtes.csv"))
df_hcp = pd.read_csv(os.path.join(current_dir, "../data/data_files/hcps.csv"))
df_regions_hcps = pd.read_csv(
    os.path.join(current_dir, "../data/data_files/regions_hcps.csv")
)
df_sales = pd.read_csv(os.path.join(current_dir, "../data/data_files/sales_train.csv"))


#
# feature: brand
#
brand_mapping = {"brand_1": 1, "brand_2": 2, "brand_3": 3}


def _brand(df_train):
    return df_train.brand.map(brand_mapping).to_numpy()


#
# feature: month (as integer)
#
months = np.sort(df_sales.month.unique())
month_mapping = dict(zip(months, range(len(months))))


def _month(df_train):
    return df_train.month.map(month_mapping).to_numpy()


#
# feature: brand12_market
#
def _brand_12_market(df_train):
    if "sales" in df_train.columns:
        df_train = df_train.drop(columns="sales")

    return (
        df_train.reset_index(
            drop=True
        )  # drop the old index and create a new sorted index
        .reset_index()  # creates a new column "index" with ascending numbers
        .merge(
            df_sales[df_sales.brand == "brand_12_market"],
            on=["month", "region"],
            how="left",
        )
        .sort_values(by="index")
        .sales.to_numpy()
    )


#
# feature: brand12_market
#
def _brand_3_market(df_train):
    if "sales" in df_train.columns:
        df_train = df_train.drop(columns="sales")

    return (
        df_train.reset_index(
            drop=True
        )  # drop the old index and create a new sorted index
        .reset_index()  # creates a new column "index" with ascending numbers
        .merge(
            df_sales[df_sales.brand == "brand_3_market"],
            on=["month", "region"],
            how="left",
        )
        .sort_values(by="index")
        .sales.to_numpy()
    )


#
# feature: brand3 sales
#
def _brand_3(df_train):
    if "sales" in df_train.columns:
        df_train = df_train.drop(columns="sales")

    return (
        df_train.reset_index(
            drop=True
        )  # drop the old index and create a new sorted index
        .reset_index()  # creates a new column "index" with ascending numbers
        .merge(
            df_sales[df_sales.brand == "brand_3"], on=["month", "region"], how="left",
        )
        .sort_values(by="index")
        .sales.to_numpy()
    )


#
# feature: mails sent to hcp with tier X
#
df_rte.time_sent = pd.to_datetime(df_rte.time_sent)
df_rte = df_rte.merge(
    df_hcp.drop(columns=["specialty", "region"]), on="hcp", how="left"
)


def _mails_from_tier(df_train, tier=1):
    mails_per_month_tier = (
        df_rte[df_rte.tier == tier]
        .groupby(
            ["brand", "region", pd.Grouper(key="time_sent", freq="MS")], dropna=False
        )
        .hcp.size()
        .reset_index()
        .rename(columns={"hcp": "mails_tier"})
    )
    mails_per_month_tier["month"] = mails_per_month_tier.time_sent.dt.strftime("%Y-%m")

    return (
        df_train.reset_index(
            drop=True
        )  # drop the old index and create a new sorted index
        .reset_index()  # creates a new column "index" with ascending numbers
        .merge(mails_per_month_tier, on=["brand", "region", "month"], how="left")
        .sort_values(by="index")
        .fillna(0)
        .mails_tier.to_numpy()
    )


#
# feature: population
#
df_regions = pd.read_csv(os.path.join(current_dir, "../data/data_files/regions.csv"))


def _population(df_train):
    mapping = dict(zip(df_regions.region, df_regions.population))
    return df_train.region.map(mapping).to_numpy()


def _area(df_train):
    mapping = dict(zip(df_regions.region, df_regions.area))
    return df_train.region.map(mapping).to_numpy()


def _pci16(df_train):
    mapping = dict(zip(df_regions.region, df_regions.pci16))
    return df_train.region.map(mapping).to_numpy()


#
# feature: content count for specialty
#
def _content_count(df_train, specialty):
    df_content_count = (
        df_rte[df_rte.specialty == specialty]
        .groupby(["region", "brand"])
        .content_id.size()
        .reset_index()
    )
    return (
        df_train.reset_index(
            drop=True
        )  # drop the old index and create a new sorted index
        .reset_index()  # creates a new column "index" with ascending numbers
        .merge(df_content_count, on=["brand", "region"], how="left")
        .sort_values(by="index")
        .content_id.fillna(0)
        .to_numpy()
    )


def _hcps_per_region(df_train, specialty):
    assert specialty in df_regions_hcps.columns

    return (
        df_train.reset_index(
            drop=True
        )  # drop the old index and create a new sorted index
        .reset_index()  # creates a new column "index" with ascending numbers
        .merge(df_regions_hcps, on=["region"], how="left")
        .sort_values(by="index")[specialty]
        .fillna(0)
        .to_numpy()
    )


#
# all features
#
feature_mapping = {
    "brand": _brand,
    "month": _month,
    "brand_12_market": _brand_12_market,
    "brand_3_market": _brand_3_market,
    "brand_3": _brand_3,
    "tier1_mail": lambda df_train: _mails_from_tier(df_train, tier=1),
    "tier2_mail": lambda df_train: _mails_from_tier(df_train, tier=2),
    "population": _population,
    "area": _area,
    "pci16": _pci16,
    "content_count_imp": lambda df_train: _content_count(
        df_train, "Internal medicine / pneumology"
    ),
    # hcps per region
    "hcp_im_per_region": lambda df_train: _hcps_per_region(
        df_train, "Internal medicine"
    ),
    "hcp_gp_per_region": lambda df_train: _hcps_per_region(
        df_train, "General practicioner"
    ),
    "hcp_imgp_per_region": lambda df_train: _hcps_per_region(
        df_train, "Internal medicine and general practicioner"
    ),
    "hcp_imp_per_region": lambda df_train: _hcps_per_region(
        df_train, "Internal medicine / pneumology"
    ),
    "hcp_p_per_region": lambda df_train: _hcps_per_region(df_train, "Pediatrician"),
}


def create_feature_matrix(df_train, features):
    """
    Usage:
        df_train = df_sales[(df_sales.region == 'region_19') & (df_sales.brand == 'brand_1')]
        features = ['brand', 'tier1_mail', 'tier2_mail']
        create_feature_matrix(df_train, features)
    """
    columns = [
        feature_mapping[feature](df_train)
        if isinstance(feature, str)
        else feature(df_train)
        for feature in features
    ]
    df = pd.DataFrame(dict(zip(features, columns)))
    return df.to_numpy()
