import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from preprocessing import load_and_merge
import joblib # 모델 저장용
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import logging

logging.getLogger('lightgbm').setLevel(logging.ERROR)

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
    '대기압(mmHg)_예측',

    # 11-1
    'hour',
    'month',
    'weekday',
    # 11-2
    '일사량x기온',
    '습도x기온',
    '일사량x절대습도'
]
targets = [
    '습도(%)_관측',
    '기온(degC)_관측',
    '대기압(mmHg)_관측'
]

X = merged_df[features]
y = merged_df[targets]

''' # 3. 모델 학습 [ 11-2까지의 학습 ]
for col in y.columns:
    print(f"\n [{col}] 모델 학습중...")
    model = LGBMRegressor(random_state=42)
    model.fit(X, y[col])    # 학습

    # 모델 저장
    model_path = f'models/model_{col}.pkl'
    joblib.dump(model, model_path)
    print(f"저장 완료: {model_path}")
'''

# 11단계 3) 하이퍼파라미터 튜닝 및 교차검증증
# 1. TimeSeriesSplit 설정
tss = TimeSeriesSplit(n_splits=3)

# 2. 하이퍼파라미터 그리드 설정
param_grid = {
    'num_leaves' : [31, 63],
    'learning_rate' : [0.1, 0.01],
    'n_estimators' : [100, 300],
    'max_depth' : [-1, 10, 20]
}

# 3. 타깃별로 GridSearchCV 수행
for col in targets:
    print(f"\n[{col}] 하이퍼파라미터 튜닝 시작...")

    # LightGBM 회귀 모델 객체
    base_model = LGBMRegressor(random_state=42, verbose=-1)

    # GridSearchCV: RMSE(neg_root_mean_squared_error) 기준
    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=tss,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )

    # Grid Search 수행
    grid.fit(X, y[col])

    # 최적 파라미터 및 교차검증 점수 확인
    best_params = grid.best_params_
    best_score = -grid.best_score_ # 음수로 반환된 값의 부호를 반전하여 RMSE로 변환
    print(f" 최적 파라미터: {best_params}")
    print(f" CV RMSE (평균): {best_score:.4f}")

    # 4. 최적 모델 저장
    best_model = grid.best_estimator_
    model_path = f'models/model_{col}.pkl'
    joblib.dump(best_model, model_path)
    print(f" 저장 완료: {model_path}")

    # 5. 전체 데이터에 대해 최종 학습 후 MAE-RMSE 출력
    best_model.fit(X, y[col])
    y_pred = best_model.predict(X)
    mae = np.mean(np.abs(y[col] - y_pred))
    rmse = mean_squared_error(y[col], y_pred) ** 0.5
    print(f" 전체 데이터 MAE: {mae:.4f}, RMSE: {rmse:.4f}")