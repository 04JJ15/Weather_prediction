from src.preprocessing import load_and_merge
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 파일 경로
forecast_path = 'data/데이터_분석과제_7_기상예측데이터_2401_2503.csv'
observed_path = 'data/데이터_분석과제_7_기상관측데이터_2401_2503.csv'

# 병합 실행
merged_df = load_and_merge(forecast_path, observed_path)

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

''' 4단계) 병합 결과 확인 및 CSV로 저장
# 결과 확인
print("\n 병합된 데이터 컬럼 목록 : ")
print(merged_df.columns.tolist())

# CSV 파일로 저장
output_path = 'data/병합_결과.csv'
merged_df.to_csv(output_path, index=False, encoding='utf-8-sig') # 엑셀 한글 호환을 위한 인코딩
 '''

''' 5단계) 결측값, 기본 통계 분석, 이상치 확인
# 1. 결측치 확인
print("\n [결측치 확인]")
null_counts = merged_df.isnull().sum()
print(null_counts[null_counts > 0] if null_counts.sum() > 0 else "결측치 없음")

# 2. 기본 통계 요약
print("\n [기본 통계 요약]")
print(merged_df.describe())

# 3. 이상치 탐색 (IQR 방식)
print("\n [이상치 탐색 - IQR 기준]")
def count_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return ((series < lower) | (series > upper)).sum()

for col in merged_df.select_dtypes(include='number').columns:
    outlier_count = count_outliers(merged_df[col])
    print(f"{col}: {outlier_count}개")
'''

# 7단계) Feature / Target 분리
X = merged_df[features]
y = merged_df[targets]

''' Feature / Target 확인
print("\n Feature(X) shape:", X.shape)
print(" Target(y) shape:", y.shape)

print("\n Feature 컬럼 목록:")
print(X.columns.tolist())

print("\n Target 컬럼 목록:")
print(y.columns.tolist())
'''

''' 8단계) 상관관계 히트맵 시각화
# 8단계
# 1. 입력(X) + 출력(y) 합쳐서 하나의 DataFrame 생성
corr_df = pd.concat([X, y], axis=1) # 두 개의 데이터 프레임을 열 기준으로 합침

# 2. 상관계수 계산 (피어슨 상관계수)
corr_matrix = corr_df.corr() # -1~1사이의 값으로 선형관계를 측정

# 3. 시각화 - 히트맵
plt.figure(figsize=(10, 8)) # 그래프 크기 설정
sns.heatmap(
    corr_matrix,    # 상관계수 행렬 입력
    annot=True,     # 각 칸에 숫자 값 표시
    cmap='coolwarm',# 색상 맵: 파랑~빨강
    fmt=".2f",      # 소수점 둘째 자리까지 표시
    linewidths=0.5  # 셀 간 경계선 두께
)
plt.title(" Feature-Target 상관관계 히트맵", fontsize=14)
plt.tight_layout()
plt.show()

'''
