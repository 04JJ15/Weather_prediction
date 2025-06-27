import pandas as pd
import numpy as np
import joblib
import optuna
from lightgbm import LGBMRegressor, early_stopping
from catboost import CatBoostRegressor, Pool, cv as cat_cv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from preprocessing import load_and_merge

N_SPLITS = 5

def objective_lgb(trial, X, y):
    # 1) LGBM 하이퍼파라미터 공간
    param = {
        'num_leaves':         trial.suggest_int('num_leaves', 16, 256),
        'learning_rate':      trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'n_estimators':       trial.suggest_int('n_estimators', 50, 500),
        'max_depth':          trial.suggest_int('max_depth', 5, 30),
        'min_child_samples':  trial.suggest_int('min_child_samples', 5, 100),
        'subsample':          trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree':   trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'lambda_l1':          trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2':          trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'min_split_gain':     trial.suggest_float('min_gain_to_split', 0.0, 1.0),
    }

    tss = TimeSeriesSplit(n_splits=N_SPLITS)
    rmses = []
    for ti, vi in tss.split(X):
        X_tr, X_va = X.iloc[ti], X.iloc[vi]
        y_tr, y_va = y.iloc[ti], y.iloc[vi]

        model = LGBMRegressor(**param, random_state=42, verbose=-1)
        # Early stopping via callback
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[early_stopping(stopping_rounds=50)]
        )
        preds = model.predict(X_va)
        rmses.append(mean_squared_error(y_va, preds) ** 0.5)

    return np.mean(rmses)


def objective_cat(trial, X, y):
    # 2) CatBoost 하이퍼파라미터 공간
    param = {
        'learning_rate':      trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        'depth':              trial.suggest_int('depth', 4, 12),
        'l2_leaf_reg':        trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'bagging_temperature':trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'border_count':       trial.suggest_int('border_count', 32, 255),
        'iterations':         trial.suggest_int('iterations', 100, 1000),
    }

    tss = TimeSeriesSplit(n_splits=N_SPLITS)
    rmses = []
    for ti, vi in tss.split(X):
        X_tr, X_va = X.iloc[ti], X.iloc[vi]
        y_tr, y_va = y.iloc[ti], y.iloc[vi]

        # CatBoost는 Pool을 이용해 early stopping
        train_pool = Pool(X_tr, y_tr)
        valid_pool = Pool(X_va, y_va)

        model = CatBoostRegressor(**param, random_state=42, verbose=False)
        model.fit(
            train_pool,
            eval_set=valid_pool,
            early_stopping_rounds=50,
            verbose=False
        )
        preds = model.predict(X_va)
        rmses.append(mean_squared_error(y_va, preds) ** 0.5)

    return np.mean(rmses)


def objective_rf(trial, X, y):
    # 3) RF 하이퍼파라미터 공간
    param = {
        'n_estimators':      trial.suggest_int('n_estimators', 100, 1000),
        'max_depth':         trial.suggest_int('max_depth', 5, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features':      trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    }

    tss = TimeSeriesSplit(n_splits=N_SPLITS)
    rmses = []
    for ti, vi in tss.split(X):
        X_tr, X_va = X.iloc[ti], X.iloc[vi]
        y_tr, y_va = y.iloc[ti], y.iloc[vi]

        model = RandomForestRegressor(**param, random_state=42, n_jobs=-1)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        rmses.append(mean_squared_error(y_va, preds) ** 0.5)

    return np.mean(rmses)


def optimize_and_save(name, objective_fn, X, y, prefix='model'):
    study = optuna.create_study(direction='minimize',
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective_fn(trial, X, y), n_trials=100)

    best_params = study.best_params
    best_rmse = study.best_value
    print(f"\n🔍 [{name}] 최적 파라미터: {best_params}")
    print(f"🔍 [{name}] CV RMSE: {best_rmse:.4f}")

    # 4) 전체 데이터 학습 및 저장
    if name == 'lgbm':
        model = LGBMRegressor(**best_params, random_state=42, verbose=-1)
    elif name == 'cat':
        model = CatBoostRegressor(**best_params, random_state=42, verbose=False)
    else:  # 'rf'
        model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)

    model.fit(X, y)
    joblib.dump(model, f"models/{prefix}_{name}_optuna.pkl")
    print(f"✅ [{name}] 모델 저장: models/{prefix}_{name}_optuna.pkl")


if __name__ == '__main__':
    df = load_and_merge(
        'data/raw/데이터_분석과제_7_기상예측데이터_2401_2503.csv',
        'data/raw/데이터_분석과제_7_기상관측데이터_2401_2503.csv'
    )

    features = [
        '일사량(w/m^2)_예측','습도(%)_예측','절대습도_예측',
        '기온(degC)_예측','대기압(mmHg)_예측',
        'hour','month','weekday',
        '일사량x기온','습도x기온','일사량x절대습도',
        'humidity_lag_1h','humidity_lag_3h','humidity_lag_6h',
        'humidity_lag_12h','humidity_lag_24h'
    ]
    X = df[features]
    y = df['습도(%)_관측']

    optimize_and_save('lgbm', objective_lgb, X, y, prefix='humidity')
    optimize_and_save('cat', objective_cat, X, y, prefix='humidity')
    optimize_and_save('rf',  objective_rf,  X, y, prefix='humidity')