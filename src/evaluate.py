import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from preprocessing import load_and_merge

def main():
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

    # 3. 평가 가중치
    weights = {
        '습도(%)_관측' : 0.3,
        '기온(degC)_관측' : 0.5,
        '대기압(mmHg)_관측' : 0.2
    }

    final_score = 0.0
    print("\n 모델 평가 시작")

    for col in targets:
        # 4. 모델 로드 & 예측
        model = joblib.load(f'models/model_{col}.pkl')
        y_pred = model.predict(X)

        # 5. MAE, RMSE 계산
        mae = mean_absolute_error(y[col], y_pred)
        rmse = mean_squared_error(y[col], y_pred)**0.5

        # 6. 가중 점수
        weighted = weights[col] * ((mae + rmse) / 2)
        final_score += weighted

        print(f"\n [{col}]")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  가중 점수: {weighted:.4f}")

    print(f"\n 최종 가중 평균 점수: {final_score:.4f}\n")

if __name__ == "__main__":
    main()
