import os
from pathlib import Path
import pandas as pd

# 기상 예측 데이터와 관측 데이터를 시간 기준으로 병합
# 파생피처 추가

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

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

     # 컬럼 순서 정리
    cols = [
        'datetime',
        '일사량(w/m^2)_예측', '습도(%)_예측', '절대습도_예측',
        '기온(degC)_예측', '대기압(mmHg)_예측',
        'hour', 'month', 'weekday',
        '일사량x기온', '습도x기온', '일사량x절대습도', '일사량x대기압',
        # 관측 타깃
        '습도(%)_관측', '기온(degC)_관측', '대기압(mmHg)_관측'
    ]
    merged = merged[cols]

    return merged

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