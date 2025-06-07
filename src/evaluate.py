import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from preprocessing import load_and_merge


def evaluate():
    # 1) 데이터 로드 & 전처리
    forecast_path = 'data/데이터_분석과제_7_기상예측데이터_2401_2503.csv'
    observed_path = 'data/데이터_분석과제_7_기상관측데이터_2401_2503.csv'
    df = load_and_merge(forecast_path, observed_path)

    # 2) 공통 Feature 정의
    base_feats = [
        '일사량(w/m^2)_예측', '습도(%)_예측', '절대습도_예측',
        '기온(degC)_예측', '대기압(mmHg)_예측',
        'hour', 'month', 'weekday',
        '일사량x기온', '습도x기온', '일사량x절대습도'
    ]
    lag_feats = [f'humidity_lag_{lag}h' for lag in [1, 3, 6, 12, 24]]

    # 3) 타깃 및 가중치
    targets = ['습도(%)_관측', '기온(degC)_관측', '대기압(mmHg)_관측']
    weights = {'습도(%)_관측': 0.3, '기온(degC)_관측': 0.5, '대기압(mmHg)_관측': 0.2}
    final_score = 0.0

    print("\n[모델 평가 시작]\n")

    for col in targets:
        # 4) 모델 불러오기 및 예측
        if col == '습도(%)_관측':
            # 앙상블 모델
            lgb_model, cat_model, rf_model = joblib.load('models/humidity_ensemble_models.pkl')
            X_eval = df[base_feats + lag_feats]
            preds = np.mean([
                lgb_model.predict(X_eval),
                cat_model.predict(X_eval),
                rf_model.predict(X_eval)
            ], axis=0)
        else:
            # 단일 Optuna 모델
            model = joblib.load(f'models/model_{col}_optuna.pkl')
            X_eval = df[base_feats]
            preds = model.predict(X_eval)

        # 5) MAE, RMSE 계산
        y_true = df[col]
        mae = mean_absolute_error(y_true, preds)
        rmse = np.sqrt(mean_squared_error(y_true, preds))

        # 6) 가중 점수
        weighted_score = weights[col] * ((mae + rmse) / 2)
        final_score += weighted_score

        print(f"[{col}]")
        print(f"  MAE:        {mae:.4f}")
        print(f"  RMSE:       {rmse:.4f}")
        print(f"  가중 점수: {weighted_score:.4f}\n")

    print(f"최종 가중 평균 점수: {final_score:.4f}\n")


if __name__ == '__main__':
    evaluate()
'''

**설명**
1. `load_and_merge()`로 **Lag 피처가 적용된** `df`를 준비합니다.
2. **습도**는 **앙상블** 모델(`humidity_ensemble_models.pkl`)을 불러와 5개 Lag 포함 16개 피처를 이용해 평균 예측합니다.
3. **기온**, **대기압**은 Optuna 최적화된 단일 모델(`model_{col}_optuna.pkl`)을 불러와 11개 기본 피처를 사용합니다.
4. 각 타깃별 MAE·RMSE를 계산하고, **공모전 가중치**를 적용해 가중 점수를 계산합니다.
5. 전체 타깃의 가중 평균 점수를 출력합니다.

이제 터미널에서 `python -m src.evaluate` 를 실행하면 공모전 최종 점수가 계산됩니다.
'''

# 10단계 11-1, 11-2, 11-3.4 단계 까지의 평가점수 그래프 그리기