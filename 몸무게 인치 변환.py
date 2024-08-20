import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import  cross_val_score
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('./data/5.HeightWeight.csv')
print(data)

data['Height(Inches)'] = data['Height(Inches)']*2.54
data['Weight(Pounds)'] = data['Weight(Pounds)']*0.453592



# 1. 파일을 불러온다
array = data.values
# 2. 불러온 파일을 열별로 나눈다
X = array[:,1]
#  [행, 열] 구조임 그래서 열을 불러올려고 두번째에 수를 넣는거임
Y = array[:,2]
# 3. 열을 나눌때 각 열번호에서 하나뺀값으로 넣어서 나눈다
plt.clf()
fig, ax = plt.subplots()
#  4. 그림그릴 준비

#  그림 초기화

# 기본으로 주어진 데이터에 대한 산점도 표
X1=X.reshape(-1,1)
#  5. 현제 데이터값이 1대 1이기 때문에 X독립변수에 대해서 행렬로 만들어 주는 함수
# ------------------------------------------------------------------------------------

(X_train,X_text,
 Y_train,Y_test)=train_test_split(X1,Y,test_size=0.001)
# 6. 기본적으로 어떤 데이터를 딥러닝 시키고 어떤데이터를 테스트용으로 뺄지 정하는 단계
model =LinearRegression()
#  선형 분석으로 모델을 만드는 것
model.fit(X_train,Y_train)

# y=ax+b 일때 a값이 머다

# b값이 머다
#  7. 그 선형 분석의 값을 6번에서 고른 데이터 값으로 만들어라는 명령
y_pred = model.predict(X_text)
#  8. 선형 분석한 값에 6번의 테스트 값을 집어 넣으라는 명령
print(y_pred)

# -------------------------------------------------------------------------
# plt.figure(figsize=(10,6))
# 9. 차트의 자체 크기를 정하는 함수
plt.scatter(X_text,Y_test,color='blue',
            marker='o')
# 10.테스트를 한 것의 원본을 점으로 표현해라
plt.plot(X_text,y_pred,color='r'
            ,marker='x')
# 11. 테스트를 한 것의 결과를 선으로 표혀해라
plt.show()

mae=mean_absolute_error(y_pred,Y_test)
print(mae)