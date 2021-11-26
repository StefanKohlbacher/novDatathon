from novDatathon.src.metric_participants import ComputeMetrics
from sklearn.model_selection import KFold
import numpy as np


def cross_validate(df_sales_train, train):
    nrFolds = 5
    folds = {}
    regions = df_sales_train[df_sales_train.brand == "brand_1"].region.unique()
    kfold = KFold(n_splits=nrFolds, shuffle=True, random_state=991)
    for i, indices in enumerate(kfold.split(regions)):
        trainIndex, testIndex = indices
        folds[i] = {
            "training": regions[trainIndex.tolist()],
            "test": regions[testIndex.tolist()],
        }

    def _f(fold):
        df_train = df_sales_train[
            df_sales_train.region.isin(fold["training"])
            & df_sales_train.brand.isin(["brand_1", "brand_2"])
        ]
        df_val = df_sales_train[
            df_sales_train.region.isin(fold["test"])
            & df_sales_train.brand.isin(["brand_1", "brand_2"])
        ]
        df_market = df_sales_train[df_sales_train.region.isin(fold["test"])]

        df_submission = train(df_train, df_val.drop(columns="sales"))

        df_val.merge(df_submission, on=["brand", "region", "month"], how="left")

        df_val = df_val[df_val.month >= "2020-07"]
        df_submission = df_submission[df_submission.month >= "2020-07"]
        df_market = df_market[df_market.month >= "2020-07"]

        return ComputeMetrics(df_submission, df_market, df_val)

    results = np.array([_f(fold) for fold in folds.values()])
    return results.mean(axis=0)
