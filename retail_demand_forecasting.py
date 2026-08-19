import os
import io
import numpy as np
import pandas as pd
import requests
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

TRAIN_URL = "https://raw.githubusercontent.com/Wang-Shuo/Kaggle-Rossman-Store-Sales/master/input/train.csv"
STORE_URL = "https://raw.githubusercontent.com/Wang-Shuo/Kaggle-Rossman-Store-Sales/master/input/store.csv"


def fetch_csv(url, local_path):
    if os.path.exists(local_path):
        return pd.read_csv(local_path, low_memory=False)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with open(local_path, "wb") as f:
        f.write(resp.content)
    return pd.read_csv(io.BytesIO(resp.content), low_memory=False)


def load_rossmann_data():
    train = fetch_csv(TRAIN_URL, "train.csv")
    store = fetch_csv(STORE_URL, "store.csv")
    df = train.merge(store, on="Store", how="left")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.rename(columns={
        "Store": "store", "Date": "date", "Sales": "sales",
        "Promo": "promo", "Open": "open", "StateHoliday": "state_holiday",
        "SchoolHoliday": "school_holiday",
    })
    df["holiday"] = ((df["state_holiday"].astype(str) != "0") | (df["school_holiday"] == 1)).astype(int)
    df = df.sort_values(["store", "date"]).reset_index(drop=True)
    return df


def add_calendar_features(df):
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["quarter"] = df["date"].dt.quarter
    return df


def build_features(df):
    df = add_calendar_features(df)
    df = df.sort_values(["store", "date"]).reset_index(drop=True)

    grp = df.groupby("store")["sales"]
    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = grp.shift(lag)

    shifted = grp.shift(1)
    for win in [7, 14, 28]:
        df[f"roll_mean_{win}"] = shifted.groupby(df["store"]).transform(lambda s: s.rolling(win).mean())
        df[f"roll_std_{win}"] = shifted.groupby(df["store"]).transform(lambda s: s.rolling(win).std())

    df["StoreType"] = df["StoreType"].astype("category").cat.codes
    df["Assortment"] = df["Assortment"].astype("category").cat.codes
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())

    df = df.dropna(subset=["lag_28", "roll_mean_28"]).reset_index(drop=True)
    return df


FEATURES = [
    "store", "promo", "holiday", "open", "dow", "month", "weekofyear",
    "is_weekend", "quarter", "StoreType", "Assortment", "CompetitionDistance",
    "lag_1", "lag_7", "lag_14", "lag_28",
    "roll_mean_7", "roll_std_7", "roll_mean_14", "roll_std_14",
    "roll_mean_28", "roll_std_28",
]
TARGET = "sales"


def time_based_split(df, test_days=42):
    max_date = df["date"].max()
    test_start = max_date - pd.Timedelta(days=test_days)
    train_df = df[df["date"] < test_start].copy()
    test_df = df[df["date"] >= test_start].copy()
    return train_df, test_df, test_start


def train_lightgbm(train_df):
    train_df = train_df.copy()
    train_df["store"] = train_df["store"].astype("category")
    cat_idx = [FEATURES.index("store")]
    lgb_train = lgb.Dataset(train_df[FEATURES], label=train_df[TARGET], categorical_feature=cat_idx)
    params = {"objective": "regression", "metric": "rmse", "learning_rate": 0.05, "num_leaves": 64, "verbose": -1}
    model = lgb.train(params, lgb_train, num_boost_round=300)
    return model


def train_catboost(train_df):
    model = CatBoostRegressor(
        iterations=200, learning_rate=0.08, depth=8,
        cat_features=["store"], loss_function="RMSE",
        verbose=False, thread_count=4,
    )
    model.fit(train_df[FEATURES], train_df[TARGET])
    return model


def wape(y_true, y_pred):
    denom = np.sum(np.abs(y_true))
    return np.sum(np.abs(y_true - y_pred)) / denom * 100 if denom > 0 else np.nan


def evaluate(test_df, test_start, lgb_model, cat_model, horizons=(7, 14, 28)):
    results = []
    test_df = test_df.copy()
    test_df["store"] = test_df["store"].astype("category")

    for h in horizons:
        sub = test_df[test_df["date"] < test_start + pd.Timedelta(days=h)]
        sub = sub[sub["open"] == 1]
        if len(sub) == 0:
            continue
        y_true = sub[TARGET].values
        naive_pred = sub["lag_7"].values
        lgb_pred = lgb_model.predict(sub[FEATURES])
        cat_pred = cat_model.predict(sub[FEATURES])

        for name, pred in [("Naive(lag7)", naive_pred), ("LightGBM", lgb_pred), ("CatBoost", cat_pred)]:
            rmse = np.sqrt(mean_squared_error(y_true, pred))
            mae = mean_absolute_error(y_true, pred)
            w = wape(y_true, pred)
            results.append({
                "horizon_days": h, "model": name,
                "RMSE": round(rmse, 2), "MAE": round(mae, 2), "WAPE_%": round(w, 2),
            })
    return pd.DataFrame(results)


def main():
    print("Step 1/5: Downloading real Rossmann Store Sales dataset (Kaggle) ...")
    raw_df = load_rossmann_data()
    print(f"  -> {raw_df.shape[0]} rows, {raw_df['store'].nunique()} stores, "
          f"{raw_df['date'].min().date()} to {raw_df['date'].max().date()}")

    print("Step 2/5: Building leakage-safe time-series features ...")
    feat_df = build_features(raw_df)
    feat_df.to_csv("rossmann_features.csv", index=False)
    print(f"  -> {feat_df.shape[0]} rows after feature engineering")

    print("Step 3/5: Time-based train/test split (last 42 days held out) ...")
    train_df, test_df, test_start = time_based_split(feat_df, test_days=42)
    print(f"  -> Train: {train_df.shape[0]} rows | Test: {test_df.shape[0]} rows | Test start: {test_start.date()}")

    print("Step 4/5: Training LightGBM and CatBoost ...")
    lgb_model = train_lightgbm(train_df)
    lgb_model.save_model("lightgbm_model.txt")
    cat_model = train_catboost(train_df)
    cat_model.save_model("catboost_model.cbm")

    print("Step 5/5: Evaluating at 7/14/28-day horizons ...")
    results_df = evaluate(test_df, test_start, lgb_model, cat_model)
    results_df.to_csv("model_comparison_results.csv", index=False)

    print("\n" + "=" * 60)
    print("FINAL RESULTS on real Rossmann data (lower = better)")
    print("=" * 60)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()