import pandas as pd

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

    # 병합 inner 조인 사용용
    merged = pd.merge(forecast, observed, on='datetime', how='inner')
    return merged

# 내가 직접 볼수 있게 코드를 제공해달라하기