import os
from pathlib import Path
import numpy as np
import pandas as pd

# 기상 예측 데이터와 관측 데이터를 시간 기준으로 병합
# 파생피처 추가

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# 실제 대회 검증 기간(1개월)을 흉내낸 홀드아웃 시작일.
# 데이터 제공 기간이 2024-01-01~2025-03-31 이므로 마지막 1개월(2025-03)을
# 어떤 학습/튜닝/스태킹에도 쓰지 않고 최종 리포트 전용으로 떼어놓는다.
HOLDOUT_START = "2025-03-01"

def load_raw(forecast_fname: str, observed_fname: str) -> pd.DataFrame:
    """원시 forecast/observed CSV 를 읽어서 병합한 뒤 반환."""
    fpath = RAW_DIR / forecast_fname
    opath = RAW_DIR / observed_fname

    # CSV 로딩
    forecast = pd.read_csv(fpath, parse_dates=['기상관측일시'])
    observed = pd.read_csv(opath, parse_dates=['기상관측일시'])
    # 컬럼명 통일
    forecast.rename(columns={'기상관측일시' : 'datetime'}, inplace=True)
    observed.rename(columns={'기상관측일시' : 'datetime'}, inplace=True)
    # 예측/관측 컬럼 구분 명시
    forecast.columns = ['datetime'] + [col + '_예측' for col in forecast.columns if col != 'datetime']
    observed.columns = ['datetime'] + [col + '_관측' for col in observed.columns if col != 'datetime']

    # 병합 inner 조인 사용
    merged = pd.merge(forecast, observed, on="datetime", how="inner")
    merged.sort_values("datetime", inplace=True)
    merged.reset_index(drop=True, inplace=True)

    # 대기압 단위 통일 hPa to mmHg: 1 hPa ~= 0.750062 mmHg
    merged["대기압(mmHg)_예측"]  = merged["대기압(hPa)_예측"]  * 0.750062
    merged.drop(columns=["대기압(hPa)_예측"], inplace=True)

    return merged

def make_features(merged: pd.DataFrame) -> pd.DataFrame:

    # 11단계 1) 시간 파생변수 추가
    merged['hour'] = merged['datetime'].dt.hour
    merged['month'] = merged['datetime'].dt.month
    merged['weekday'] = merged['datetime'].dt.weekday

    # 11단계 2) 교호작용 피처 생성
    merged['일사량x기온'] = merged['일사량(w/m^2)_예측'] * merged['기온(degC)_예측']
    merged['습도x기온'] = merged['습도(%)_예측'] * merged['기온(degC)_예측']
    merged['일사량x절대습도'] = merged['일사량(w/m^2)_예측'] * merged['절대습도_예측']
    merged['일사량x대기압'] = merged['일사량(w/m^2)_예측'] * merged['대기압(mmHg)_예측']

    # 습도 모델 보강 피처: 이슬점(Magnus-Tetens), VPD(Tetens), 일사량 로그변환.
    # 예측 컬럼만으로 계산하므로 미래 시점에도 그대로 사용 가능(리크 없음).
    T = merged['기온(degC)_예측']
    RH = merged['습도(%)_예측']
    a, b = 17.62, 243.12
    alpha = np.log(RH / 100.0) + (a * T) / (b + T)
    merged['이슬점_예측'] = (b * alpha) / (a - alpha)

    es = 0.6108 * np.exp(17.27 * T / (T + 237.3))  # 포화 수증기압(kPa)
    ea = es * RH / 100.0
    merged['VPD_예측'] = es - ea

    merged['일사량log_예측'] = np.log1p(merged['일사량(w/m^2)_예측'])

    # MOS식 잔차(오차) 타깃: 모델은 절대값이 아니라 "관측-예측" 오차를 학습한다.
    merged['습도(%)_오차'] = merged['습도(%)_관측'] - merged['습도(%)_예측']
    merged['기온(degC)_오차'] = merged['기온(degC)_관측'] - merged['기온(degC)_예측']
    merged['대기압(mmHg)_오차'] = merged['대기압(mmHg)_관측'] - merged['대기압(mmHg)_예측']

     # 컬럼 순서 정리
    cols = [
        'datetime',
        '일사량(w/m^2)_예측', '습도(%)_예측', '절대습도_예측',
        '기온(degC)_예측', '대기압(mmHg)_예측',
        '이슬점_예측', 'VPD_예측', '일사량log_예측',
        'hour', 'month', 'weekday',
        '일사량x기온', '습도x기온', '일사량x절대습도', '일사량x대기압',
        # 관측 타깃(리포팅용 원본값)
        '습도(%)_관측', '기온(degC)_관측', '대기압(mmHg)_관측',
        # MOS 잔차 타깃(학습용)
        '습도(%)_오차', '기온(degC)_오차', '대기압(mmHg)_오차',
    ]
    merged = merged[cols]

    return merged


def split_holdout(df: pd.DataFrame, holdout_start: str = HOLDOUT_START):
    """실제 대회 검증 기간을 흉내낸 시간 기반 홀드아웃 분리.

    holdout_start 이후 데이터는 어떤 학습/튜닝/스태킹에도 쓰지 않고
    최종 리포트(evaluate)에서만 사용한다.
    """
    # 인덱스를 보존한다(reset하지 않음): 홀드아웃 lag 피처를 만들 때
    # train 구간의 최근 이력과 concat한 뒤 원래 인덱스로 다시 골라내야 하므로,
    # 두 조각의 인덱스가 겹치면 안 된다.
    cutoff = pd.Timestamp(holdout_start)
    df_train = df[df['datetime'] < cutoff]
    df_holdout = df[df['datetime'] >= cutoff]
    return df_train, df_holdout

def load_and_merge(
    forecast_fname: str  = "데이터_분석과제_7_기상예측데이터_2401_2503.csv",
    observed_fname: str  = "데이터_분석과제_7_기상관측데이터_2401_2503.csv",
    force_rebuild: bool  = False
) -> pd.DataFrame:
    """
    1) data/raw 에서 원본 읽어 병합  
    2) data/processed/merged.csv 에 저장된 게 있으면 재사용  
    3) make_features() 로 파생피처 생성  
    """
    merged_path = PROCESSED_DIR / "merged.csv"

    if merged_path.exists() and not force_rebuild:
        merged = pd.read_csv(merged_path, parse_dates=["datetime"])
    else:
        merged = load_raw(forecast_fname, observed_fname)
        merged.to_csv(merged_path, index=False)

    # 피처 엔지니어링
    merged = make_features(merged)
    return merged