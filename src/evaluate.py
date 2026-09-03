import os
import yaml
from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.preprocessing import load_and_merge, split_holdout
from src.features import build_lagged
from src.ensemble import StackingPipeline  # noqa: F401  (joblib.load 시 클래스 참조 필요)

CONFIG_PATH = Path(os.environ.get("PIPELINE_CONFIG", "config/targets.yaml"))
MODELS_DIR = Path(os.environ.get("PIPELINE_MODELS_DIR", "models"))

WEIGHTS = {
    '습도(%)_관측': 0.3,
    '기온(degC)_관측': 0.5,
    '대기압(mmHg)_관측': 0.2,
}


def _evaluate_split(df_source, cfg_all, split_name, restrict_index=None):
    """df_source에 lag 피처를 만든 뒤, restrict_index가 주어지면 그 행들만 남겨서
    평가한다(홀드아웃처럼 lag 계산엔 앞선 구간의 이력이 필요하지만, 실제 평가는
    holdout 구간에서만 하고 싶을 때 사용).
    """
    records = []
    final_score = 0.0
    print(f"\n▶ [{split_name}] 평가")

    for target, cfg in cfg_all.items():
        obs_col      = cfg['obs_col']
        forecast_col = cfg['forecast_col']
        target_col   = cfg['target_col']  # 예측오차(잔차)
        features     = cfg['features']
        weight       = WEIGHTS[obs_col]

        df_eval = build_lagged(df_source, cfg)
        if restrict_index is not None:
            df_eval = df_eval.loc[df_eval.index.intersection(restrict_index)]
        df_eval = df_eval.dropna(subset=features + [target_col, obs_col, forecast_col])

        # 스태킹 파이프라인 로드 & 예측 (오차 스케일)
        pipe_path = MODELS_DIR / f"{target}_stacking_pipeline.pkl"
        stack_pipe = joblib.load(pipe_path)

        X_eval = df_eval[features]
        mask = X_eval.notna().all(axis=1)
        X_eval = X_eval.loc[mask]

        pred_resid = stack_pipe.predict(X_eval)
        # 절대값 복원: 예측치(forecast_col) + 예측된 오차
        pred_abs = pred_resid + df_eval.loc[mask, forecast_col].to_numpy()
        y_true = df_eval.loc[mask, obs_col]

        mae  = mean_absolute_error(y_true, pred_abs)
        rmse = mean_squared_error(y_true, pred_abs) ** 0.5
        wscore = weight * ((mae + rmse) / 2)
        final_score += wscore

        print(f"  {obs_col}  n={len(y_true)}  MAE={mae:.4f}  RMSE={rmse:.4f}  가중점수={wscore:.4f}")

        records.append({
            "split":          split_name,
            "target_col":     obs_col,
            "y_true":         y_true.tolist(),
            "y_pred":         pred_abs.tolist(),
            "MAE":            mae,
            "RMSE":           rmse,
            "weighted_score": wscore,
        })

    print(f"  [{split_name}] 최종 가중 평균 점수: {final_score:.4f}")
    return records


def evaluate():
    cfg_all = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    df = load_and_merge(force_rebuild=False)
    df_train, df_holdout = split_holdout(df)

    print("\n[모델 평가 시작]")

    # (a) 학습기간 재적합 진단 — base 모델이 실제로 본 데이터라 낙관적일 수 있으므로
    #     "최종 성적"이 아니라 과적합 갭 확인용 참고 지표로만 사용한다.
    train_records = _evaluate_split(df_train, cfg_all, "train_fit")

    # (b) 홀드아웃(2025-03) — 어떤 학습/튜닝/스태킹에도 쓰이지 않은 구간.
    #     lag 피처 계산에는 train 구간의 최근 이력이 필요하므로 결합 후 생성하고,
    #     실제 평가 행은 holdout 구간으로만 제한한다.
    combined = pd.concat([df_train, df_holdout])
    holdout_records = _evaluate_split(
        combined, cfg_all, "holdout", restrict_index=df_holdout.index
    )

    headline = sum(r["weighted_score"] for r in holdout_records)
    print(f"\n★ 헤드라인(홀드아웃) 최종 가중 평균 점수: {headline:.4f}\n")

    return pd.DataFrame(train_records + holdout_records)


if __name__ == '__main__':
    evaluate()
