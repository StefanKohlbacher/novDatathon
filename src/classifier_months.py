import numpy as np
from .features import create_feature_matrix


def train_clf(df_train, features, clf):
    X_train = create_feature_matrix(df_train, features)
    y_train = df_train.sales.to_numpy()

    clf.fit(X_train, y_train)

    return clf


def train_and_eval_clf(
    df_train, df_val, features, clf, ci="weak_estimators",
):
    """
    Usage:
        features = ['brand', 'month', 'tier1_mail', 'tier2_mail']
        train_and_eval_rf(
            df_train, df_val,
            features=['brand', 'month', 'tier1_mail', 'tier2_mail'],
            clf=RandomForestRegressor(n_estimators=500))
    """
    clf = train_clf(df_train, features, clf)

    X_val = create_feature_matrix(df_val, features)
    y_val = clf.predict(X_val)

    df_val = df_val.copy()
    df_val["sales"] = y_val

    if ci == "simple_quantiles":
        df_val["lower"] = df_train.sales.quantile(0.1)
        df_val["upper"] = df_train.sales.quantile(0.9)
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
        df_val = df_val.merge(
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
        df_val = df_val.merge(
            upper_train_quantiles[grouping_columns + ["upper"]],
            on=grouping_columns,
            how="left",
        )
    elif ci == "weak_estimators":  # only works with random forest
        predictions = np.array(
            [estimator.predict(X_val) for estimator in clf.estimators_]
        ).T
        df_val["lower"] = np.quantile(predictions, 0.1, axis=1)
        df_val["upper"] = np.quantile(predictions, 0.9, axis=1)

    return df_val
