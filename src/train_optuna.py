import pandas as pd
import numpy as np
import joblib
import optuna
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from preprocessing import load_and_merge

# 공통 설정
CV_SPLITS = 5
FEATURES = [
    '일사량(w/m^2)_예측', '습도(%)_예측', '절대습도_예측',
    '기온(degC)_예측', '대기압(mmHg)_예측',
    'hour', 'month', 'weekday',
    '일사량x기온', '습도x기온', '일사량x절대습도',
    'humidity_lag_1h', 'humidity_lag_3h', 'humidity_lag_6h', 'humidity_lag_12h', 'humidity_lag_24h'
]
TARGET = '습도(%)_관측'

def get_cv():
    return TimeSeriesSplit(n_splits=CV_SPLITS)

# LightGBM 하이퍼파라미터 탐색

def objective_lgb(trial, X, y):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 16, 256),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
    }
    tss = get_cv()
    rmses = []
    for train_idx, val_idx in tss.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = LGBMRegressor(**params, random_state=42, verbose=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
    return np.mean(rmses)

# CatBoost 하이퍼파라미터 탐색

def objective_cat(trial, X, y):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'iterations': trial.suggest_int('iterations', 100, 500),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0)
    }
    tss = get_cv()
    rmses = []
    for train_idx, val_idx in tss.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = CatBoostRegressor(**params, random_state=42, verbose=0)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
    return np.mean(rmses)

# RandomForest 하이퍼파라미터 탐색

def objective_rf(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20)
    }
    tss = get_cv()
    rmses = []
    for train_idx, val_idx in tss.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = RandomForestRegressor(**params, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmses.append(np.sqrt(mean_squared_error(y_val, preds)))
    return np.mean(rmses)

# 최적화 및 저장

def optimize_and_save(model_name, objective_fn, X, y, n_trials=50):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda tr: objective_fn(tr, X, y), n_trials=n_trials)
    print(f"[{model_name}] Best params: {study.best_params}")
    print(f"[{model_name}] Best CV RMSE: {study.best_value:.4f}")
    # 최종 학습 및 저장
    if model_name == 'lgb':
        model = LGBMRegressor(**study.best_params, random_state=42, verbose=-1)
    elif model_name == 'cat':
        model = CatBoostRegressor(**study.best_params, random_state=42, verbose=0)
    else:
        model = RandomForestRegressor(**study.best_params, random_state=42)
    model.fit(X, y)
    joblib.dump(model, f"models/model_{model_name}_humidity_optuna.pkl")
    print(f"Saved: models/model_{model_name}_humidity_optuna.pkl")

# 메인 실행

def main():
    df = load_and_merge(
        'data/데이터_분석과제_7_기상예측데이터_2401_2503.csv',
        'data/데이터_분석과제_7_기상관측데이터_2401_2503.csv'
    )
    X = df[FEATURES]
    y = df[TARGET]
    optimize_and_save('lgb', objective_lgb, X, y, n_trials=50)
    optimize_and_save('cat', objective_cat, X, y, n_trials=50)
    optimize_and_save('rf', objective_rf, X, y, n_trials=50)

if __name__ == '__main__':
    main()