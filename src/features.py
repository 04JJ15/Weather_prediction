import pandas as pd


def build_lagged(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """cfg['lag_col'](예측치)와 cfg['target_col'](오차) 두 소스에 대해
    cfg['lag']/cfg['rolling'] 윈도우로 lag/rolling 피처를 생성해 반환한다.

    train_models.py(fold별 objective, 전체 재학습), ensemble.py(OOF),
    evaluate.py(최종 평가)에서 중복 구현되던 로직을 하나로 통합한 것.
    rolling은 항상 shift(1) 이후 계산해 자기 자신을 보지 않도록 하고,
    min_periods=1 + ffill로 시계열 앞부분 NaN을 안전하게 채운다.
    """
    out = df.copy()
    lag_windows = cfg.get("lag", [])
    roll_windows = cfg.get("rolling", [])
    sources = dict.fromkeys([cfg["lag_col"], cfg["target_col"]])  # 순서 보존 중복 제거

    for source in sources:
        for lag in lag_windows:
            out[f"{source}_lag_{lag}h"] = out[source].shift(lag)
        for w in roll_windows:
            out[f"{source}_roll_{w}h"] = (
                out[source]
                .shift(1)
                .rolling(window=w, min_periods=1)
                .mean()
                .ffill()
            )

    return out
