import pandas as pd
import os
from novDatathon.src.classifier_months import train_and_eval_clf
from novDatathon.src.cross_validate import cross_validate

current_dir = os.path.dirname(os.path.abspath(__file__))

df_submission = pd.read_csv(os.path.join(current_dir, "../data/submission_sample.csv"))
df_submission.drop(columns=["sales", "lower", "upper"], inplace=True)

df_sales_train = pd.read_csv(
    os.path.join(current_dir, "../data/data_files/sales_train.csv")
)
df_sales_train = df_sales_train[df_sales_train.brand.isin(["brand_1", "brand_2"])]


def create_submission(
    features, clf, ci, goal="sales", brand3_data=False, filename="upload_novartis.csv"
):
    """
    Usage: 
        create_submission(
            features=["brand", "month", "tier1_mail", "tier2_mail"],
            clf=RandomForestRegressor(n_estimators=200, min_samples_split=2),
            ci="weak_estimators"
        )
    """

    def train(df_train, df_val):
        return train_and_eval_clf(
            df_train,
            df_val,
            features=features,
            clf=clf,
            ci=ci,
            goal=goal,
            brand3_data=brand3_data,
        )

    print(cross_validate(train))

    df_submission_final = train(df_sales_train, df_submission)

    df_submission_final.sort_values(["month", "region", "brand"], inplace=True)
    df_submission_final = df_submission_final[df_submission_final.month >= "2020-07"]
    df_submission_final.to_csv(filename, index=False)
    return df_submission_final
