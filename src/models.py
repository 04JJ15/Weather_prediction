from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor


def build_estimator(model_name: str, params: dict):
    """model_name('lgbm'|'cat'|'rf'|'extra_trees')과 파라미터 dict로
    학습되지 않은(fresh) 회귀 모델 인스턴스를 만든다.

    train_models.py(CV objective, 최종 재학습)와 ensemble.py(폴드별 OOF 재적합)
    양쪽에서 중복 구현되던 모델 생성 분기를 통합한 것.
    """
    if model_name == "lgbm":
        return LGBMRegressor(**params, random_state=42, verbose=-1)
    elif model_name == "cat":
        return CatBoostRegressor(**params, random_state=42, verbose=False)
    elif model_name == "rf":
        return RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    elif model_name == "extra_trees":
        return ExtraTreesRegressor(**params, random_state=42, n_jobs=-1)
    raise ValueError(f"Unknown model_name: {model_name}")
