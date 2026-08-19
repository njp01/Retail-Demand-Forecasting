# Retail-Demand-Forecasting

Leakage-safe, multi-horizon retail demand forecasting pipeline built on the Kaggle
Rossmann Store Sales dataset. Compares a naive baseline against LightGBM and CatBoost
under strict time-based (expanding-window) validation.

Dataset
Source: Kaggle Rossmann Store Sales competition data (auto-downloaded by the script — no manual download needed).

Scale: 1,017,209 rows across 1,115 real German drugstore stores.

Period: January 1, 2013 to July 31, 2015.

Files used: train.csv (daily sales, promo, holiday, open/close flags) merged with store.csv (store type, assortment, competition distance, Promo2).

Pipeline Overview
Data acquisition — load_rossmann_data() downloads and merges train.csv + store.csv directly from source, caches locally, and derives a combined holiday flag from state and school holidays.

Feature engineering — build_features() creates:

Calendar features: day-of-week, month, week-of-year, quarter, weekend flag.

Lag features: sales 1, 7, 14, 28 days prior.

Rolling statistics: 7/14/28-day rolling mean and std, computed strictly on shifted (past-only) data to avoid leakage.

Store metadata: encoded StoreType, Assortment, and CompetitionDistance (missing values imputed with median).

Time-based split — time_based_split() holds out the last 42 days as test data with no shuffling, preventing any future information from leaking into training.

Model training

train_lightgbm() — LightGBM regressor with native categorical handling for store.

train_catboost() — CatBoost regressor with native categorical handling for store.

Evaluation — evaluate() scores Naive (lag-7), LightGBM, and CatBoost at 7/14/28-day forecast horizons using RMSE, MAE, and WAPE (Weighted Absolute Percentage Error), restricted to days the store was open.
