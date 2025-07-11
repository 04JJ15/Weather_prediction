# 📊 분석 코드 - 기상 예측 오차 보정

이 프로젝트는 기상청의 예측 데이터를 바탕으로 실제 관측값(습도, 기온, 대기압)의 오차를 보정하기 위한 머신러닝 분석을 수행합니다.

---

## 📁 디렉토리 구조

```
.
├── main.py
├── config/
│   └── targets.yaml
├── src/
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── ensemble.py
│   └── evaluate.py
├── data/
│   └── raw/
│       ├── 기상예측데이터_2401_2503.csv
│       └── 기상관측데이터_2401_2503.csv
├── models/           # 학습된 모델 저장
├── outputs/          # 결과 시각화 및 점수 저장
│   ├── scores/
│   └── figures/
├── requirements.txt
└── README.md
```

---

## 🚀 실행 방법

아래 명령어 하나로 전체 분석 흐름을 자동으로 실행합니다:

```bash
python main.py
```

기존 병합 데이터를 무시하고 강제로 다시 실행하고 싶다면:

```bash
python main.py --force-rebuild
```

---

## 📦 주요 실행 흐름

1. **데이터 병합 및 피처 엔지니어링**  (`src/preprocessing.py`)
2. **개별 모델 학습 및 저장**  (`src/train_models.py`)
3. **스태킹 앙상블 모델 구성**  (`src/ensemble.py`)
4. **성능 평가 및 시각화 출력**  (`src/evaluate.py`)

---

## 🛠️ 사용 라이브러리

- pandas, numpy  
- scikit-learn  
- matplotlib  
- joblib  
- optuna  
- pyyaml  
- lightgbm  
- catboost

필요한 패키지는 `requirements.txt`로 설치 가능합니다:

```bash
pip install -r requirements.txt
```

---

## 📂 출력 결과 예시

- `outputs/scores/final_scores.csv` : MAE, RMSE, 가중 점수 등 요약
- `outputs/figures/*.png` : 산점도, 중요도 그래프

---