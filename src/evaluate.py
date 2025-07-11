import yaml
from pathlib import Path
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.preprocessing import load_and_merge
from src.ensemble import StackingPipeline


def evaluate():
    # 1) 설정 읽기 및 데이터 로드
    cfg_all = yaml.safe_load(Path("config/targets.yaml").read_text(encoding="utf-8"))
    # force_rebuild=False 로 processed 캐시 사용
    df = load_and_merge(force_rebuild=False)

    records = []
    # 가중치 설정
    weights = {
        '습도(%)_관측': 0.3,
        '기온(degC)_관측': 0.5,
        '대기압(mmHg)_관측': 0.2
    }
    final_score = 0.0

    print("\n[모델 평가 시작]\n")
    # 각 타겟별 평가
    for target, cfg in cfg_all.items():
        target_col = cfg['target_col']
        weight = weights[target_col]
        print(f"▶ 평가 대상: {target_col}")

        # 2) 전체 데이터에 lag/rolling 피처 생성
        lag_windows  = cfg.get('lag', [])
        roll_windows = cfg.get('rolling', [])
        lag_source   = cfg.get('lag_col', target_col)
        features     = cfg['features']
        df_eval = df.copy()
        # lag
        for l in lag_windows:
            df_eval[f"{lag_source}_lag_{l}h"] = df_eval[lag_source].shift(l)
        # rolling (shift 1 후)
        for w in roll_windows:
            df_eval[f"{lag_source}_roll_{w}h"] = (
                df_eval[lag_source]
                    .shift(1)
                    .rolling(window=w)
                    .mean()
            )
        df_eval.dropna(subset=features + [target_col], inplace=True)

        # 3) 모델 로드 및 예측
        # 스태킹 파이프라인
        pipe_path = Path("models") / f"{target}_stacking_pipeline.pkl"
        stack_pipe = joblib.load(pipe_path)
        # 입력 칼럼
        feature_cols = cfg['features']
        X_eval = df_eval[feature_cols]
        # NA 제거
        mask = X_eval.notna().all(axis=1)
        X_eval = X_eval.loc[mask]
        y_true = df_eval.loc[mask, target_col]

        # 예측
        preds = stack_pipe.predict(X_eval)

        # 4) MAE, RMSE
        mae  = mean_absolute_error(y_true, preds)
        rmse = mean_squared_error(y_true, preds) ** 0.5

        # 5) 가중 점수
        wscore = weight * ((mae + rmse) / 2)
        final_score += wscore

        print(f"  MAE:        {mae:.4f}")
        print(f"  RMSE:       {rmse:.4f}")
        print(f"  가중 점수: {wscore:.4f}\n")

        records.append({
            "target_col":     target_col,
            "y_true":         y_true.tolist(),
            "y_pred":         preds.tolist(),
            "MAE":            mae,
            "RMSE":           rmse,
            "weighted_score": wscore
        })
        

    print(f"최종 가중 평균 점수: {final_score:.4f}\n")

    return pd.DataFrame(records)

if __name__ == '__main__':
    evaluate()

