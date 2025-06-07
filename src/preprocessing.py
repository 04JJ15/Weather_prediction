import pandas as pd
import numpy as np

# 기상 예측 데이터와 관측 데이터를 시간 기준으로 병합

# 예측값과 실제값을 한줄로 정렬

def load_and_merge(forecast_path, observed_path):

    # CSV 로딩
    forecast = pd.read_csv(forecast_path, parse_dates=['기상관측일시'])
    observed = pd.read_csv(observed_path, parse_dates=['기상관측일시'])
    # 컬럼명 통일
    forecast.rename(columns={'기상관측일시' : 'datetime'}, inplace=True)
    observed.rename(columns={'기상관측일시' : 'datetime'}, inplace=True)
    # 예측/관측 컬럼 구분 명시
    forecast.columns = ['datetime'] + [col + '_예측' for col in forecast.columns if col != 'datetime']
    observed.columns = ['datetime'] + [col + '_관측' for col in observed.columns if col != 'datetime']
    # 병합 inner 조인 사용
    merged = pd.merge(forecast, observed, on='datetime', how='inner')
    
    # 대기압 단위 통일 hPa to mmHg: 1 hPa ~= 0.750062 mmHg
    merged['대기압(mmHg)_예측'] = merged['대기압(hPa)_예측'] * 0.750062
    merged.drop(columns=['대기압(hPa)_예측'], inplace=True)

    # 11단계 1) 시간 파생변수 추가
    merged['hour'] = merged['datetime'].dt.hour
    merged['month'] = merged['datetime'].dt.month
    merged['weekday'] = merged['datetime'].dt.weekday

    # 11단계 2) 교호작용 피처 생성
    merged['일사량x기온'] = merged['일사량(w/m^2)_예측'] * merged['기온(degC)_예측']
    merged['습도x기온'] = merged['습도(%)_예측'] * merged['기온(degC)_예측']
    merged['일사량x절대습도'] = merged['일사량(w/m^2)_예측'] * merged['절대습도_예측']

    # 11단계 3.4) Lag 피처 추가 (습도 예측 시차)
    for lag in [1, 3, 6, 12, 24]:
        merged[f'humidity_lag_{lag}h'] = merged['습도(%)_예측'].shift(lag)
    
    # 결측치 제거 (Lag로 인한 NaN 행 삭제)
    merged.dropna(subset=[f'humidity_lag_{lag}h' for lag in [1, 3, 6, 12, 24]], inplace=True)

    # 컬럼 순서 정리
    cols = [
        'datetime',
        '일사량(w/m^2)_예측', '습도(%)_예측', '절대습도_예측',
        '기온(degC)_예측', '대기압(mmHg)_예측',
        'hour', 'month', 'weekday',
        '일사량x기온', '습도x기온', '일사량x절대습도',
        *[f'humidity_lag_{lag}h' for lag in [1,3,6,12,24]],
        # 관측 타깃
        '습도(%)_관측', '기온(degC)_관측', '대기압(mmHg)_관측'
    ]
    merged = merged[cols]
    
    return merged