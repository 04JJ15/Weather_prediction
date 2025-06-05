import pandas as pd
import numpy as np
import joblib
import optuna
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from preprocessing import load_and_merge

def objective(trial, X, y):
    """
    Optuna가 각 Trial마다 호출하여, 주어진 파라미터로 CV RMSE를 계산하는 함수.
    - trial: Optuna trial 객체
    - X: 피처(설명 변수) DataFrame
    - y: 타깃(종속 변수) Series
    """
    # 1) 탐색할 하이퍼파라미터 공간 정의
    param = {
        'num_leaves': trial.suggest_int('num_leaves', 16, 256),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1e-1),
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 30),
        # 추가적으로 유용한 파라미터를 더 탐색하도록 포함할 수 있음
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
    }

    # 2) 시계열 교차검증 분할기 설정 (max_train_size 미사용 → 확장형 윈도우)
    tss = TimeSeriesSplit(n_splits=5)

    rmses = []
    for train_idx, valid_idx in tss.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        # 3) 모델 학습 (CV 내 검증용 eval_set은 제거)
        model = LGBMRegressor(**param, random_state=42, verbose=-1)
        model.fit(X_train, y_train)

        # 4) 검증 세트 예측 → RMSE 계산
        preds = model.predict(X_valid)
        rmses.append(mean_squared_error(y_valid, preds) ** 0.5)

    # 5) 다섯 개 Fold RMSE 평균 반환
    return np.mean(rmses)


def optimize_target(target_name, X, y):
    """
    특정 타깃(target_name)에 대해 Optuna Study를 생성하고 최적 파라미터 탐색 후 모델 저장
    - target_name: 타깃 컬럼명 (문자열)
    - X: 피처 DataFrame
    - y: 타깃 Series
    """
    # 1) Optuna Study 생성
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=50)

    # 2) 최적 파라미터 및 CV RMSE 출력
    best_params = study.best_params
    best_rmse = study.best_value
    print(f"\n🔍 [{target_name}] 최적 파라미터: {best_params}")
    print(f"🔍 [{target_name}] CV RMSE (평균): {best_rmse:.4f}")

    # 3) 최적 모델 학습 & 저장
    best_model = LGBMRegressor(**best_params, random_state=42)
    best_model.fit(X, y)  # 전체 데이터에 대해 최종 학습
    model_path = f"models/model_{target_name}_optuna.pkl"
    joblib.dump(best_model, model_path)
    print(f"✅ [{target_name}] 최적 모델 저장 완료: {model_path}")


def main():
    # 1) 데이터 로드 및 전처리
    forecast_path = 'data/데이터_분석과제_7_기상예측데이터_2401_2503.csv'
    observed_path = 'data/데이터_분석과제_7_기상관측데이터_2401_2503.csv'
    merged_df = load_and_merge(forecast_path, observed_path)

    # 2) 피처 & 각 타깃 분리 (시간 파생 + 교호작용 포함 완료된 상태)
    features = [
        '일사량(w/m^2)_예측',
        '습도(%)_예측',
        '절대습도_예측',
        '기온(degC)_예측',
        '대기압(mmHg)_예측',
        'hour', 
        'month', 
        'weekday',
        '일사량x기온', 
        '습도x기온', 
        '일사량x절대습도'
    ]
    X = merged_df[features]
    targets = {
        '습도(%)_관측': merged_df['습도(%)_관측'],
        '기온(degC)_관측': merged_df['기온(degC)_관측'],
        '대기압(mmHg)_관측': merged_df['대기압(mmHg)_관측']
    }

    # 3) 타깃별 최적화 반복
    for target_name, y in targets.items():
        print(f"\n========== [{target_name}] 최적화 시작 ==========")
        optimize_target(target_name, X, y)


if __name__ == "__main__":
    main()