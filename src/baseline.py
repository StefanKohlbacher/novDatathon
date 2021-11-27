import pandas as pd


def train_baseline(df_train, df_val, q=0.1):
    mean_sales_brand_1 = (
        df_train[df_train.brand == "brand_1"].groupby("month").sales.mean()
    )
    lower_brand_1 = (
        df_train[df_train.brand == "brand_1"].groupby("month").sales.quantile(0.1)
    )
    upper_brand_1 = (
        df_train[df_train.brand == "brand_1"].groupby("month").sales.quantile(1 - q)
    )

    mean_sales_brand_2 = (
        df_train[df_train.brand == "brand_2"].groupby("month").sales.mean()
    )
    lower_brand_2 = (
        df_train[df_train.brand == "brand_2"].groupby("month").sales.quantile(0.1)
    )
    upper_brand_2 = (
        df_train[df_train.brand == "brand_2"].groupby("month").sales.quantile(1 - q)
    )

    def dummy_submission_region(region, brand):
        if brand == "brand_1":
            return pd.DataFrame(
                dict(
                    region=region,
                    brand=brand,
                    sales=mean_sales_brand_1,
                    lower=lower_brand_1,
                    upper=upper_brand_1,
                )
            ).reset_index()
        else:
            return pd.DataFrame(
                dict(
                    region=region,
                    brand=brand,
                    sales=mean_sales_brand_2,
                    lower=lower_brand_2,
                    upper=upper_brand_2,
                )
            ).reset_index()

    df_submission = pd.concat(
        [
            dummy_submission_region(region, brand)
            for region in df_val.region.unique()
            for brand in ["brand_1", "brand_2"]
        ]
    )

    df_submission.sort_values(["month", "region", "brand"], inplace=True)

    return df_submission
