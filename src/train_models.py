# src/train_models.py

import argparse
import yaml
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor, early_stopping
from catboost import CatBoostRegressor, Pool
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from preprocessing import load_and_merge  # src 폴더에서 실행할 때

def objective_factory(model_name, space, df, X_cols, y_col,
                      lag_windows, roll_windows, lag_source,
                      n_splits=5):
    """
    model_name: 'lgbm'|'cat'|'rf'|'extra_tree'
    space: optuna_space dict from YAML
    df:    전체 병합된 DataFrame
    X_cols: 피처 컬럼 리스트 (lag/rolling 포함)
    y_col: 관측 타깃 컬럼명
    lag_windows, roll_windows: YAML 에서 읽은 리스트
    lag_source: lag/rolling 생성에 사용할 원본 컬럼명 (예: '습도(%)_예측')
    """
    def _objective(trial):
        # 1) 파라미터 샘플링
        if model_name == 'lgbm':
            param = {
                'num_leaves':         trial.suggest_int('num_leaves',         *space['num_leaves']),
                'learning_rate':      trial.suggest_float('learning_rate',    *space['learning_rate'], log=True),
                'n_estimators':       trial.suggest_int('n_estimators',       *space['n_estimators']),
                'max_depth':          trial.suggest_int('max_depth',          *space['max_depth']),
                'min_child_samples':  trial.suggest_int('min_child_samples', *space.get('min_child_samples', [5,100])),
                'subsample':          trial.suggest_float('subsample',       *space.get('subsample', [0.5,1.0])),
                'colsample_bytree':   trial.suggest_float('colsample_bytree',*space.get('colsample_bytree', [0.5,1.0])),
                'lambda_l1':          trial.suggest_float('lambda_l1',       *space['lambda_l1'], log=True),
                'lambda_l2':          trial.suggest_float('lambda_l2',       *space['lambda_l2'], log=True),
                'min_gain_to_split':  trial.suggest_float('min_gain_to_split', *space['min_gain_to_split']),
            }
        elif model_name == 'cat':
            param = {
                'learning_rate':      trial.suggest_float('learning_rate',    *space['learning_rate'], log=True),
                'depth':              trial.suggest_int('depth',              *space['depth']),
                'l2_leaf_reg':        trial.suggest_float('l2_leaf_reg',      *space['l2_leaf_reg'], log=True),
                'bagging_temperature':trial.suggest_float('bagging_temperature', *space['bagging_temperature']),
                'border_count':       trial.suggest_int('border_count',       *space['border_count']),
                'iterations':         trial.suggest_int('iterations',         *space['iterations']),
            }
        elif model_name == 'rf':
            param = {
                'n_estimators':       trial.suggest_int('n_estimators',       *space['n_estimators']),
                'max_depth':          trial.suggest_int('max_depth',          *space['max_depth']),
                'min_samples_split':  trial.suggest_int('min_samples_split',  *space['min_samples_split']),
                'min_samples_leaf':   trial.suggest_int('min_samples_leaf',   *space['min_samples_leaf']),
                'max_features':       trial.suggest_categorical('max_features',    space['max_features']),
            }
        else:  # 'extra_tree'
            param = {
                'n_estimators':       trial.suggest_int('n_estimators',       *space['n_estimators']),
                'max_depth':          trial.suggest_int('max_depth',          *space['max_depth']),
                'min_samples_split':  trial.suggest_int('min_samples_split',  *space['min_samples_split']),
                'min_samples_leaf':   trial.suggest_int('min_samples_leaf',   *space['min_samples_leaf']),
                'max_features':       trial.suggest_categorical('max_features',    space['max_features']),
            }

        # 2) TimeSeriesSplit fold-wise CV
        tss = TimeSeriesSplit(n_splits=n_splits)
        rmses = []
        for tr_idx, va_idx in tss.split(df):
            train = df.iloc[tr_idx].copy()
            val   = df.iloc[va_idx].copy()

            # (a) train에서만 lag/rolling 생성
            for lag in lag_windows:
                train[f"{lag_source}_lag_{lag}h"] = train[lag_source].shift(lag)
            for w in roll_windows:
                train[f"{lag_source}_roll_{w}h"] = (
                    train[lag_source]
                    .shift(1)
                    .rolling(window=w, min_periods=1)  
                    .mean()
                    .ffill()                          
                )
            train.dropna(subset=X_cols + [y_col], inplace=True)

            # (b) val에는 train 기준으로 계산된 값만 사용
            combined = pd.concat([train, val])
            for lag in lag_windows:
                combined[f"{lag_source}_lag_{lag}h"] = combined[lag_source].shift(lag)
            for w in roll_windows:
                combined[f"{lag_source}_roll_{w}h"] = (
                    combined[lag_source]
                    .shift(1)
                    .rolling(window=w, min_periods=1)
                    .mean()
                    .ffill()
                )
            val = combined.loc[val.index].copy()
            val.dropna(subset=X_cols + [y_col], inplace=True)

            X_tr, y_tr = train[X_cols], train[y_col]
            X_va, y_va = val[X_cols],   val[y_col]

            # (c) 모델 학습
            if model_name == 'lgbm':
                m = LGBMRegressor(**param, random_state=42, verbose=-1)
                m.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    callbacks=[early_stopping(stopping_rounds=50)]
                )
                preds = m.predict(X_va, num_iteration=m.best_iteration_)
                
            elif model_name == 'cat':
                pool_tr = Pool(X_tr, y_tr)
                pool_va = Pool(X_va, y_va)
                m = CatBoostRegressor(**param, random_state=42, verbose=False)
                m.fit(pool_tr, eval_set=pool_va,
                      early_stopping_rounds=50, verbose=False)
                best_it = m.get_best_iteration()
                preds = m.predict(X_va, ntree_end=best_it)
                
            elif model_name == 'rf':
                m = RandomForestRegressor(**param, random_state=42, n_jobs=-1)
                m.fit(X_tr, y_tr)
                preds = m.predict(X_va)
                
            else:  # extra_tree
                m = ExtraTreesRegressor(**param, random_state=42, n_jobs=-1)
                m.fit(X_tr, y_tr)
                preds = m.predict(X_va)
                
            rmses.append(mean_squared_error(y_va, preds) ** 0.5)

        return np.mean(rmses)

    return _objective


def optimize_and_save(target, model_name, objective_fn, X, y, prefix):
    study = optuna.create_study(direction='minimize',
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective_fn, n_trials=100)

    best_params = study.best_params
    best_rmse   = study.best_value
    print(f"[{target} | {model_name}] CV RMSE: {best_rmse:.4f}")
    print(f"[{target} | {model_name}] Best params: {best_params}")

    # 전체 데이터 재학습 및 저장
    if model_name == 'lgbm':
        final_model = LGBMRegressor(**best_params, random_state=42, verbose=-1)
    elif model_name == 'cat':
        final_model = CatBoostRegressor(**best_params, random_state=42, verbose=False)
    elif model_name == 'rf':
        final_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    else:
        final_model = ExtraTreesRegressor(**best_params, random_state=42, n_jobs=-1)

    final_model.fit(X, y)
    out_path = Path("models") / f"{prefix}_{model_name}_optuna.pkl"
    joblib.dump(final_model, out_path)
    print(f"✅ Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["humidity","temperature","pressure"],
                        default="humidity")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="raw → processed CSV 재생성 여부")
    args = parser.parse_args()

    # 1) config 로드
    cfg    = yaml.safe_load(Path("config/targets.yaml").read_text(encoding="utf-8"))
    conf_t = cfg[args.target]
    features    = conf_t["features"]
    lag_windows = conf_t.get("lag", [])
    roll_windows= conf_t.get("rolling", [])
    models_cfg  = conf_t["models"]
    y_col       = conf_t["target_col"]
    lag_source  = conf_t.get("lag_col", y_col)

    # 2) 데이터 로드
    df = load_and_merge(force_rebuild=args.force_rebuild)

    # 3) 전체 데이터에 lag/rolling 미리 생성 (final fit 시점용)
    for lag in lag_windows:
        df[f"{lag_source}_lag_{lag}h"] = df[lag_source].shift(lag)
    for w in roll_windows:
        df[f"{lag_source}_roll_{w}h"] = (
            df[lag_source].shift(1)
                         .rolling(window=w)
                         .mean()
        )
    df.dropna(subset=features + [y_col], inplace=True)

    X = df[features]
    y = df[y_col]

    # 4) 각 모델별 Optuna 실행
    for model_name, mconf in models_cfg.items():
        space = mconf["optuna_space"]
        print(f"\n▶▶▶ {args.target} | {model_name} optuna_space:", space)
        obj = objective_factory(
            model_name, space, df, features, y_col,
            lag_windows, roll_windows, lag_source,
            n_splits=5
        )
        optimize_and_save(args.target, model_name, obj, X, y, prefix=args.target)


if __name__ == "__main__":
    main()
