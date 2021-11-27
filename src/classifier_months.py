import numpy as np
import pandas as pd
from .features import create_feature_matrix
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

df_sales = pd.read_csv(os.path.join(current_dir, "../data/data_files/sales_train.csv"))


def create_training_set(
    df_train, goal="sales", brand3_data=False,
):
    if brand3_data:
        df_brand3 = df_sales[df_sales.brand == "brand_3"]
        df_train = pd.concat([df_train, df_brand3])

    if goal == "diff_sales":
        if not brand3_data:
            df_train = df_train.copy()
        df_train["date"] = pd.to_datetime(df_train.month, format="%Y-%m")
        df_train.sort_values(["brand", "region", "date"], inplace=True)
        df_train["diff_sales"] = (
            df_train.groupby(["brand", "region"]).sales.diff().fillna(0)
        )
    return df_train


def train_clf(df_train, features, clf, goal):
    X_train = create_feature_matrix(df_train, features)
    y_train = df_train[goal].to_numpy()

    clf.fit(X_train, y_train)

    return clf


def train_and_eval_clf(
    df_train,
    df_val,
    features,
    clf,
    ci="weak_estimators",
    goal="sales",
    brand3_data=False,
):
    """
    Usage:
        features = ['brand', 'month', 'tier1_mail', 'tier2_mail']
        train_and_eval_rf(
            df_train, df_val,
            features=['brand', 'month', 'tier1_mail', 'tier2_mail'],
            clf=RandomForestRegressor(n_estimators=500))
    """
    df_train = create_training_set(df_train, goal=goal, brand3_data=brand3_data)

    clf = train_clf(df_train, features, clf, goal)

    X_val = create_feature_matrix(df_val, features)
    y_val = clf.predict(X_val)

    df_result = df_val.copy()
    df_result[goal] = y_val

    if ci == "simple_quantiles":
        df_result["lower"] = df_train.sales.quantile(0.1)
        df_result["upper"] = df_train.sales.quantile(0.9)
    elif (
        ci == "quantiles_brand_month" or ci == "quantiles_month"
    ):  # quantiles_brand_month best so far
        if ci == "quantiles_brand_month":
            grouping_columns = ["brand", "month"]
        else:
            grouping_columns = ["month"]

        lower_train_quantiles = (
            df_train.groupby(grouping_columns)
            .sales.quantile(0.1)
            .reset_index()
            .rename(columns={"sales": "lower"})
        )
        df_result = df_result.merge(
            lower_train_quantiles[grouping_columns + ["lower"]],
            on=grouping_columns,
            how="left",
        )
        upper_train_quantiles = (
            df_train.groupby(grouping_columns)
            .sales.quantile(0.9)
            .reset_index()
            .rename(columns={"sales": "upper"})
        )
        df_result = df_result.merge(
            upper_train_quantiles[grouping_columns + ["upper"]],
            on=grouping_columns,
            how="left",
        )
    elif ci == "weak_estimators":  # only works with random forest
        predictions = np.array(
            [estimator.predict(X_val) for estimator in clf.estimators_]
        ).T
        df_result["lower"] = np.quantile(predictions, 0.1, axis=1)
        df_result["upper"] = np.quantile(predictions, 0.9, axis=1)

    # convert diff sales to sales if necessary
    if goal == "diff_sales":
        df_result["date"] = pd.to_datetime(df_result.month, format="%Y-%m")
        df_result.sort_values(["brand", "region", "date"], inplace=True)
        df_result["sales"] = df_result.groupby(["brand", "region"]).diff_sales.cumsum()

        df_result.lower = df_result.sales + df_result.lower
        df_result.upper = df_result.sales + df_result.upper

        df_result.lower = df_result.lower.clip(lower=0)

    return df_result
