import os
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error

from src.preprocessing import load_and_merge, split_holdout
from src.features import build_lagged
from src.models import build_estimator

CONFIG_PATH = Path(os.environ.get("PIPELINE_CONFIG", "config/targets.yaml"))
MODELS_DIR = Path(os.environ.get("PIPELINE_MODELS_DIR", "models"))


class StackingPipeline:
    """
    Meta model stacking pipeline: takes dict of base models and a trained meta model.
    """
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model

    def predict(self, X):
        # X: DataFrame of original features including fold-wise generated lag/rolling columns
        preds = []
        for name, mdl in self.base_models.items():
            preds.append(mdl.predict(X))
        mat = np.column_stack(preds)
        return self.meta_model.predict(mat)


def _load_best_params(target, model_names):
    return {
        name: yaml.safe_load(
            (MODELS_DIR / f"{target}_{name}_best_params.yaml").read_text(encoding="utf-8")
        )
        for name in model_names
    }


def generate_oof(df, cfg):
    """진짜 OOF 예측 생성.

    기존 구현은 전체 데이터로 이미 재학습된(=검증 폴드를 이미 본) 모델을 그대로
    불러와 val fold를 예측했기 때문에 리키지가 있었다. 여기서는 train_models.py가
    저장해둔 best_params로 폴드마다 새 모델을 fit해서, val fold를 한 번도 보지
    않은 상태에서 예측한다.
    """
    feat_cols = cfg["features"]
    y_col     = cfg["target_col"]
    model_names = list(cfg["models"].keys())
    best_params = _load_best_params(cfg["name"], model_names)

    oof_df = pd.DataFrame(index=df.index, columns=model_names, dtype=float)

    tss = TimeSeriesSplit(n_splits=5)
    for tr_idx, va_idx in tss.split(df):
        train_raw = df.iloc[tr_idx].copy()
        val_raw   = df.iloc[va_idx].copy()

        # 1) train에만 lag/rolling 생성
        train = build_lagged(train_raw, cfg)
        train.dropna(subset=feat_cols + [y_col], inplace=True)

        # 2) val에도 동일 방식(train 기준 과거값만) 적용
        combined = build_lagged(pd.concat([train_raw, val_raw]), cfg)
        val = combined.loc[val_raw.index].copy()
        val.dropna(subset=feat_cols + [y_col], inplace=True)

        X_tr, y_tr = train[feat_cols], train[y_col]
        X_va = val[feat_cols]

        # 3) 폴드 전용으로 새로 fit -> 진짜 OOF
        for name in model_names:
            m = build_estimator(name, best_params[name])
            m.fit(X_tr, y_tr)
            oof_df.loc[val.index, name] = m.predict(X_va)

    return oof_df

def main():
    cfg_all = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    df_all = load_and_merge()
    # 홀드아웃(2025-03)은 스태킹 학습에도 절대 사용하지 않는다.
    df_train, _df_holdout = split_holdout(df_all)

    for target, cfg in cfg_all.items():
        cfg["name"] = target
        print(f"\n▶ Generating stacking pipeline for [{target}]")

        # 2) 진짜 OOF 예측 생성
        oof_df = generate_oof(df_train, cfg)

        # 3) Base 모델별 OOF RMSE 출력 (NaN은 제외, 오차 스케일 기준)
        y_true = df_train[cfg["target_col"]]
        for name in cfg["models"]:
            y_pred = oof_df[name]
            mask   = y_pred.notna()
            rmse   = mean_squared_error(y_true[mask], y_pred[mask]) ** 0.5
            print(f"[OOF] Base `{name}` RMSE(진짜 OOF): {rmse:.4f}")

        # 4) 메타 모델 학습을 위한 mask (모두 예측된 행만)
        mask   = oof_df.notna().all(axis=1)
        meta_X = oof_df.loc[mask].values.astype(float)
        meta_y = df_train.loc[mask, cfg["target_col"]].values

        # 5) 메타 모델 (시계열 CV용 Ridge)
        meta_model = RidgeCV(
            alphas=[0.1, 1.0, 10.0],
            scoring="neg_root_mean_squared_error",
            cv=TimeSeriesSplit(n_splits=5)
        )
        meta_model.fit(meta_X, meta_y)
        meta_pred = meta_model.predict(meta_X)
        meta_rmse = mean_squared_error(meta_y, meta_pred) ** 0.5
        print(f"[OOF] Stacking meta-model RMSE: {meta_rmse:.4f}")

        # 6) 배포용 파이프라인 저장: base model은 학습기간 전체(df_train)로
        #    재학습된 pkl을 그대로 사용해 실사용 예측 시 정보량을 최대화한다.
        base_models = {
            name: joblib.load(MODELS_DIR / f"{target}_{name}_optuna.pkl")
            for name in cfg["models"]
        }
        pipe = StackingPipeline(base_models, meta_model)
        out_path = MODELS_DIR / f"{target}_stacking_pipeline.pkl"
        joblib.dump(pipe, out_path)
        print(f"✅ Saved stacking pipeline: {out_path}")


if __name__ == "__main__":
    main()
