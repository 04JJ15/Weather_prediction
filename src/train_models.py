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
from lightgbm import early_stopping
from catboost import Pool

from src.preprocessing import load_and_merge, split_holdout
from src.features import build_lagged
from src.models import build_estimator


def _sample_params(trial, model_name, space):
    if model_name == 'lgbm':
        return {
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
        return {
            'learning_rate':      trial.suggest_float('learning_rate',    *space['learning_rate'], log=True),
            'depth':              trial.suggest_int('depth',              *space['depth']),
            'l2_leaf_reg':        trial.suggest_float('l2_leaf_reg',      *space['l2_leaf_reg'], log=True),
            'bagging_temperature':trial.suggest_float('bagging_temperature', *space['bagging_temperature']),
            'border_count':       trial.suggest_int('border_count',       *space['border_count']),
            'iterations':         trial.suggest_int('iterations',         *space['iterations']),
        }
    else:  # 'rf' | 'extra_trees' -> 동일 스키마
        return {
            'n_estimators':       trial.suggest_int('n_estimators',       *space['n_estimators']),
            'max_depth':          trial.suggest_int('max_depth',          *space['max_depth']),
            'min_samples_split':  trial.suggest_int('min_samples_split',  *space['min_samples_split']),
            'min_samples_leaf':   trial.suggest_int('min_samples_leaf',   *space['min_samples_leaf']),
            'max_features':       trial.suggest_categorical('max_features',    space['max_features']),
        }


def _fit_predict(model_name, param, X_tr, y_tr, X_va, y_va):
    m = build_estimator(model_name, param)
    if model_name == 'lgbm':
        m.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[early_stopping(stopping_rounds=50)]
        )
        preds = m.predict(X_va, num_iteration=m.best_iteration_)
    elif model_name == 'cat':
        pool_tr = Pool(X_tr, y_tr)
        pool_va = Pool(X_va, y_va)
        m.fit(pool_tr, eval_set=pool_va, early_stopping_rounds=50, verbose=False)
        preds = m.predict(X_va, ntree_end=m.get_best_iteration())
    else:  # rf | extra_trees
        m.fit(X_tr, y_tr)
        preds = m.predict(X_va)
    return preds


def objective_factory(model_name, space, df, X_cols, y_col, cfg, n_splits=5):
    """
    model_name: 'lgbm'|'cat'|'rf'|'extra_trees'
    space: optuna_space dict from YAML
    df:    타깃별 학습 구간 DataFrame (홀드아웃 제외)
    X_cols: 피처 컬럼 리스트 (lag/rolling 포함)
    y_col: 학습 타깃(오차) 컬럼명
    cfg:   config/targets.yaml의 해당 타깃 설정 dict (lag_col/target_col/lag/rolling 포함)
    """
    def _objective(trial):
        param = _sample_params(trial, model_name, space)

        tss = TimeSeriesSplit(n_splits=n_splits)
        rmses = []
        for fold_idx, (tr_idx, va_idx) in enumerate(tss.split(df)):
            train_raw = df.iloc[tr_idx].copy()
            val_raw   = df.iloc[va_idx].copy()

            # (a) train에서만 lag/rolling 생성
            train = build_lagged(train_raw, cfg)
            train.dropna(subset=X_cols + [y_col], inplace=True)

            # (b) val에는 train 기준(과거)으로 계산된 값만 사용
            combined = build_lagged(pd.concat([train_raw, val_raw]), cfg)
            val = combined.loc[val_raw.index].copy()
            val.dropna(subset=X_cols + [y_col], inplace=True)

            X_tr, y_tr = train[X_cols], train[y_col]
            X_va, y_va = val[X_cols],   val[y_col]

            preds = _fit_predict(model_name, param, X_tr, y_tr, X_va, y_va)
            fold_rmse = mean_squared_error(y_va, preds) ** 0.5
            rmses.append(fold_rmse)

            # 실제 pruning 동작: 폴드별 중간 결과를 보고하고 가지치기 판단
            trial.report(float(np.mean(rmses)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(rmses)

    return _objective


def optimize_and_save(target, model_name, objective_fn, X, y, prefix, n_trials=100):
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='minimize',
                                 sampler=sampler,
                                 pruner=optuna.pruners.MedianPruner())
    study.optimize(objective_fn, n_trials=n_trials)

    best_params = study.best_params
    best_rmse   = study.best_value
    print(f"[{target} | {model_name}] CV RMSE(오차 기준): {best_rmse:.4f}")
    print(f"[{target} | {model_name}] Best params: {best_params}")

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    # ensemble.py가 폴드별로 새로 fit해서 진짜 OOF를 만들 때 재사용할 best_params 저장
    params_path = models_dir / f"{prefix}_{model_name}_best_params.yaml"
    params_path.write_text(
        yaml.safe_dump(best_params, allow_unicode=True), encoding="utf-8"
    )
    print(f"✅ Saved: {params_path}")

    # 배포용(전체 학습기간 재학습) 모델 저장 — 실사용 예측에 최대 정보 활용
    final_model = build_estimator(model_name, best_params)
    final_model.fit(X, y)
    out_path = models_dir / f"{prefix}_{model_name}_optuna.pkl"
    joblib.dump(final_model, out_path)
    print(f"✅ Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["humidity","temperature","pressure"],
                        default="humidity")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="raw → processed CSV 재생성 여부")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Optuna trial 수 (스모크 테스트 시 축소해서 사용)")
    args = parser.parse_args()

    # 1) config 로드
    cfg    = yaml.safe_load(Path("config/targets.yaml").read_text(encoding="utf-8"))
    conf_t = cfg[args.target]
    features   = conf_t["features"]
    models_cfg = conf_t["models"]
    y_col      = conf_t["target_col"]

    # 2) 데이터 로드 + 홀드아웃(2025-03) 완전 배제
    df = load_and_merge(force_rebuild=args.force_rebuild)
    df_train, _df_holdout = split_holdout(df)

    # 3) 학습 구간 전체에 lag/rolling 미리 생성 (final fit 시점용)
    df_train = build_lagged(df_train, conf_t)
    df_train.dropna(subset=features + [y_col], inplace=True)

    X = df_train[features]
    y = df_train[y_col]

    # 4) 각 모델별 Optuna 실행
    for model_name, mconf in models_cfg.items():
        space = mconf["optuna_space"]
        print(f"\n▶▶▶ {args.target} | {model_name} optuna_space:", space)
        obj = objective_factory(
            model_name, space, df_train, features, y_col, conf_t, n_splits=5
        )
        optimize_and_save(args.target, model_name, obj, X, y,
                           prefix=args.target, n_trials=args.n_trials)


if __name__ == "__main__":
    main()
