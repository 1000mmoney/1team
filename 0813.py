# matplotlib 설치
import matplotlib.pyplot as plt

#pandas 설치
import pandas as pd

# numpy 설치
import numpy as np

# scikit-learn 설치
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 각 항목 이름 붙이기
colums = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv('./data/pima-indians-diabetes.data.csv', names=colums)

# 바꾸기 할 항목 범위 정하기
array = data.values
X = array[:, 0:8]
Y = array[:, 8]

# 소수점 아래로 바꾸기 1, 데이터 전처리 : Min-Max 스케일링
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_X = scaler.fit_transform(X)

# 데이터 분할
X_train, X_test, Y_train, Y_test = train_test_split(rescaled_X, Y, test_size=0.3)

# 모델 선택 및 학습
model = LinearRegression()
model.fit(X_train, Y_train)

# 학습된 모델을 사용하여 테스트 데이터에 대한 예측 수행
y_pred = model.predict(X_test)
print(y_pred)

# 예측값을 0 또는 1로 변환 (임계값 설정 필요)
y_pred_binary = (y_pred > 0.5).astype(int)
print(y_pred_binary)

# 예측 정확도 계산
accuracy = accuracy_score(Y_test, y_pred_binary)
print(accuracy)

df_Y_test = pd.DataFrame(Y_test)
df_Y_pred_binary = pd.DataFrame(y_pred_binary)
df_Y_test.to_csv("./results/y_test.csv")
df_Y_pred_binary.to_csv("./results/y_pred.csv")