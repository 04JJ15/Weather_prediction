import joblib
import numpy as np
import pandas as pd
import catboost
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from src.preprocessing import load_and_merge


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


def generate_oof(df, cfg):
    lag_windows  = cfg.get("lag", [])
    roll_windows = cfg.get("rolling", [])
    lag_source   = cfg["lag_col"]
    feat_cols    = cfg["features"]

    oof_df = pd.DataFrame(index=df.index, columns=cfg["models"].keys())

    tss = TimeSeriesSplit(n_splits=5)
    for name in cfg["models"]:
        mdl = joblib.load(f"models/{cfg['name']}_{name}_optuna.pkl")
        for tr_idx, va_idx in tss.split(df):
            train = df.iloc[tr_idx].copy()
            val   = df.iloc[va_idx].copy()

            # 1) train에만 lag/roll 생성
            for lag in lag_windows:
                train[f"{lag_source}_lag_{lag}h"] = train[lag_source].shift(lag)
            for w in roll_windows:
                train[f"{lag_source}_roll_{w}h"] = (
                    train[lag_source].shift(1)
                                     .rolling(w, min_periods=1)
                                     .mean()
                                     .ffill()
                )
            train.dropna(subset=feat_cols + [cfg["target_col"]], inplace=True)

            # 2) val에도 동일 방식 적용
            combined = pd.concat([train, val])
            for lag in lag_windows:
                combined[f"{lag_source}_lag_{lag}h"] = combined[lag_source].shift(lag)
            for w in roll_windows:
                combined[f"{lag_source}_roll_{w}h"] = (
                    combined[lag_source].shift(1)
                                      .rolling(w, min_periods=1)
                                      .mean()
                                      .ffill()
                )
            val = combined.loc[val.index].copy()
            val.dropna(subset=feat_cols + [cfg["target_col"]], inplace=True)

            # 3) 예측 및 OOF 기록
            X_va = val[feat_cols]
            preds = mdl.predict(X_va)
            oof_df.loc[val.index, name] = preds

    return oof_df

def main():
    import yaml

    cfg_all = yaml.safe_load(Path("config/targets.yaml").read_text(encoding="utf-8"))
    df_all = load_and_merge()

    for target, cfg in cfg_all.items():
        cfg["name"] = target
        print(f"\n▶ Generating stacking pipeline for [{target}]")

        # 2) OOF 예측 생성
        oof_df = generate_oof(df_all, cfg)

        # 3) Base 모델별 OOF RMSE 출력 (NaN은 제외)
        y_true = df_all[cfg["target_col"]]
        for name in cfg["models"]:
            y_pred = oof_df[name]
            mask   = y_pred.notna()
            rmse   = mean_squared_error(y_true[mask], y_pred[mask]) ** 0.5
            print(f"[OOF] Base `{name}` RMSE: {rmse:.4f}")

        # 4) 메타 모델 학습을 위한 mask (모두 예측된 행만)
        mask   = oof_df.notna().all(axis=1)
        meta_X = oof_df.loc[mask].values
        meta_y = df_all.loc[mask, cfg["target_col"]].values

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

        # 4) 파이프라인 저장
        base_models = {
            name: joblib.load(Path("models") / f"{target}_{name}_optuna.pkl")
            for name in cfg["models"]
        }
        pipe = StackingPipeline(base_models, meta_model)
        out_path = Path("models") / f"{target}_stacking_pipeline.pkl"
        joblib.dump(pipe, out_path)
        print(f"✅ Saved stacking pipeline: {out_path}")


if __name__ == "__main__":
    main()
