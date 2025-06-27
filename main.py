# main.py
import os
import sys
import argparse
import yaml
import joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.preprocessing import load_and_merge
from src.train_models import main as train_models_main
from src.ensemble import main as ensemble_main
from src.evaluate import evaluate  # 여러분이 완성한 evaluate() 함수

# ----------------------------------------
def ensure_dirs():
    for sub in ["figures", "scores"]:
        d = Path("outputs") / sub
        d.mkdir(parents=True, exist_ok=True)

def save_scores(df_scores: pd.DataFrame):
    path = Path("outputs/scores/final_scores.csv")
    df_scores.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"✅ Scores saved to {path}")

def plot_and_save(y_true, y_pred, target_col):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    plt.rc('font', family=['Malgun Gothic', 'DejaVu Sans'])
    plt.figure(figsize=(6,4))
    plt.scatter(y_true, y_pred, alpha=0.3, s=5)
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             'r--')
    plt.title(f"{target_col}: True vs Pred")
    plt.xlabel("True")
    plt.ylabel("Pred")
    fpath = Path("outputs/figures") / f"{target_col.replace('/','_')}_scatter.png"
    plt.tight_layout()
    plt.savefig(fpath)
    plt.close()
    print(f"✅ Figure saved to {fpath}")

def run_full_pipeline(force_rebuild=False):
    original_argv = sys.argv.copy()
    
    try:
        # 1) 학습 & 모델 저장 (각 --target)
        for tgt in ["humidity", "temperature", "pressure"]:
            print(f"\n===== Training target: {tgt} =====")
            # sys.argv를 train_models_main이 기대하는 형태로 설정
            sys.argv = [original_argv[0], "--target", tgt]
            if force_rebuild:
                sys.argv.append("--force-rebuild")
            train_models_main()  # 이제 내부에서 sys.argv를 파싱해 동작합니다.
    finally:
        # 원래 argv로 복원
        sys.argv = original_argv
    

    # 2) 앙상블 파이프라인 생성
    print("\n===== Generating stacking pipelines =====")
    ensemble_main()

    # 3) 최종 평가 & 결과 수집
    print("\n===== Final Evaluation =====")
    df_scores = evaluate()  
    # evaluate()는 targets 별 MAE/RMSE/가중점수를 DataFrame 으로 리턴한다고 가정

    # 4) 결과 저장 & 시각화
    save_scores(df_scores)
    # scatter plot 예시
    for col in df_scores["target_col"]:
        # 실제값/예측값을 evaluate()에서 반환해주면
        row = df_scores[df_scores["target_col"] == col].iloc[0]
        y_true = row["y_true"]   # 이젠 순수한 리스트 [0.1,0.2,…]
        y_pred = row["y_pred"]
        plot_and_save(y_true, y_pred, col)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-rebuild", action="store_true",
                        help="processed 캐시 파일 재생성 여부")
    args = parser.parse_args()

    ensure_dirs()
    run_full_pipeline(force_rebuild=args.force_rebuild)

