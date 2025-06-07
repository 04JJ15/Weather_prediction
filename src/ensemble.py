import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from preprocessing import load_and_merge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# ---------------------------
# 1) 전체 데이터로 학습된 모델 불러오기 및 최적 파라미터 추출
# ---------------------------
# Optuna로 최적화된 LGBM 모델
lgb_final = joblib.load('models/model_lgb_humidity_optuna.pkl')
all_params_lgb = lgb_final.get_params()
best_params_lgb = {k: all_params_lgb[k] for k in [
    'num_leaves','learning_rate','n_estimators','max_depth',
    'min_child_samples','subsample','colsample_bytree'
]}

# Optuna로 최적화된 CatBoost 모델
cat_final = joblib.load('models/model_cat_humidity_optuna.pkl')
all_params_cat = cat_final.get_params()
best_params_cat = {k: all_params_cat[k] for k in [
    'learning_rate','depth','iterations','l2_leaf_reg','bagging_temperature'
]}

# Optuna로 최적화된 RandomForest 모델
rf_final = joblib.load('models/model_rf_humidity_optuna.pkl')
all_params_rf = rf_final.get_params()
best_params_rf = {k: all_params_rf[k] for k in [
    'n_estimators','max_depth','min_samples_split','min_samples_leaf'
]}

# ---------------------------
# 2) 앙상블 CV 평가 및 저장 함수
# ---------------------------
def train_ensemble(forecast_path, observed_path, save_path):
    # 데이터 로드 & 전처리 (Lag 포함된 상태)
    df = load_and_merge(forecast_path, observed_path)
    features = [
        '일사량(w/m^2)_예측','습도(%)_예측','절대습도_예측',
        '기온(degC)_예측','대기압(mmHg)_예측',
        'hour','month','weekday',
        '일사량x기온','습도x기온','일사량x절대습도',
        'humidity_lag_1h','humidity_lag_3h','humidity_lag_6h',
        'humidity_lag_12h','humidity_lag_24h'
    ]
    X = df[features]
    y = df['습도(%)_관측']

    # TimeSeriesSplit 설정
    tss = TimeSeriesSplit(n_splits=5)
    rmses = []

    # 각 Fold마다 재학습 후 앙상블 예측
    for train_idx, val_idx in tss.split(X):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        # LGBM fold 재학습
        lgb = LGBMRegressor(**best_params_lgb, random_state=42, verbose=-1)
        lgb.fit(X_tr, y_tr)
        # CatBoost fold 재학습
        cat = CatBoostRegressor(**best_params_cat, random_state=42, verbose=0)
        cat.fit(X_tr, y_tr)
        # RF fold 재학습
        rf = RandomForestRegressor(**best_params_rf, random_state=42)
        rf.fit(X_tr, y_tr)

        # 앙상블 예측 및 RMSE
        preds = np.mean([
            lgb.predict(X_va),
            cat.predict(X_va),
            rf.predict(X_va)
        ], axis=0)
        rmses.append(mean_squared_error(y_va, preds) ** 0.5)

    print(f"[Ensemble CV RMSE] 평균: {np.mean(rmses):.4f}")

    # ---------------------------
    # 3) 최종 모델 저장
    # ---------------------------
    # 전체 데이터로 학습된 모델을 그대로 저장
    joblib.dump((lgb_final, cat_final, rf_final), save_path)
    print(f"✅ 앙상블 모델 저장: {save_path}")


if __name__ == '__main__':
    train_ensemble(
        'data/데이터_분석과제_7_기상예측데이터_2401_2503.csv',
        'data/데이터_분석과제_7_기상관측데이터_2401_2503.csv',
        'models/humidity_ensemble_models.pkl'
    )