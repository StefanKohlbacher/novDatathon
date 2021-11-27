import pandas as pd
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

#
# feature: brand
#
brand_mapping = {"brand_1": 1, "brand_2": 2, "brand_3": 3}


def _brand(df_train):
    return df_train.brand.map(brand_mapping).to_numpy()


#
# feature: month (as integer)
#
df_sales = pd.read_csv(os.path.join(current_dir, "../data/data_files/sales_train.csv"))
months = np.sort(df_sales.month.unique())
month_mapping = dict(zip(months, range(len(months))))


def _month(df_train):
    return df_train.month.map(month_mapping).to_numpy()


#
# feature: mails sent to hcp with tier X
#
df_rte = pd.read_csv(os.path.join(current_dir, "../data/data_files/rtes.csv"))
df_hcp = pd.read_csv(os.path.join(current_dir, "../data/data_files/hcps.csv"))

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
        df_train.merge(
            mails_per_month_tier, on=["brand", "region", "month"], how="left"
        )
        .fillna(0)["mails_tier"]
        .to_numpy()
    )


#
# all features
#
feature_mapping = {
    "brand": _brand,
    "month": _month,
    "tier1_mail": lambda df_train: _mails_from_tier(df_train, tier=1),
    "tier2_mail": lambda df_train: _mails_from_tier(df_train, tier=2),
}


def create_feature_matrix(df_train, features):
    """
    Usage:
        df_train = df_sales[(df_sales.region == 'region_19') & (df_sales.brand == 'brand_1')]
        features = ['brand', 'tier1_mail', 'tier2_mail']
        create_feature_matrix(df_train, features)
    """
    columns = [feature_mapping[feature](df_train) for feature in features]
    df = pd.DataFrame(dict(zip(features, columns)))
    return df.to_numpy()
