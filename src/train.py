import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from preprocessing import load_and_merge
import joblib # 모델 저장용

# 1. 데이터 불러오기
forecast_path = 'data/데이터_분석과제_7_기상예측데이터_2401_2503.csv'
observed_path = 'data/데이터_분석과제_7_기상관측데이터_2401_2503.csv'
merged_df = load_and_merge(forecast_path, observed_path)

# 2. Feature / Target 분리

features = [
    '일사량(w/m^2)_예측',
    '습도(%)_예측',
    '절대습도_예측',
    '기온(degC)_예측',
    '대기압(mmHg)_예측'
]
targets = [
    '습도(%)_관측',
    '기온(degC)_관측',
    '대기압(mmHg)_관측'
]

X = merged_df[features]
y = merged_df[targets]

# 3. 모델 학습
for col in y.columns:
    print(f"\n [{col}] 모델 학습중...")
    model = LGBMRegressor(random_state=42)
    model.fit(X, y[col])    # 학습

    # 모델 저장
    model_path = f'models/model_{col}.pkl'
    joblib.dump(model, model_path)
    print(f"저장 완료: {model_path}")