import random
import numpy as np
import os

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import lightgbm as lgb
        lgb.basic._Config({'seed': seed})
    except ImportError:
        pass
    try:
        from catboost import CatBoostRegressor
        # CatBoost는 모델 인자에 random_seed 지정
    except ImportError:
        pass
    # scikit-learn의 경우, 각 모델에 random_state 파라미터를 넣어 주면 됩니다.
