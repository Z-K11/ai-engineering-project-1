import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix,accuracy_score
import sklearn.metrics as metrics
dataframe = pd.read_csv('./csv/Weather_Data.csv')
print(dataframe.head())
df_processed = pd.get_dummies(data=dataframe,columns=['RainToday', 'WindGustDir', 'WindDir9am', 'WindDir3pm'])
print(df_processed.head())
df_processed.replace(['No','Yes'],[0,1],inplace=True)
print(df_processed)
df_processed.drop('Date',axis=1,inplace=True)
df_processed=df_processed.astype(float)
print(df_processed)
feature=df_processed.drop('RainTomorrow',axis=1)
y=df_processed['RainTomorrow']
print(y.head())
x_train,x_test,y_train,y_test = train_test_split(feature,y,test_size=0.2,random_state=10)
LinearReg=LinearRegression()
LinearReg.fit(x_train,y_train)
predictions=LinearReg.predict(x_test)
LinearReg_MAE=np.mean(np.absolute(predictions-y_test))
LinearReg_MSE=metrics.mean_squared_error(y_test,predictions)
LinearReg_R2=metrics.r2_score(y_test,predictions)
print(LinearReg_MAE)
print(LinearReg_MSE)
print(LinearReg_R2)
report=pd.DataFrame({'Metrics ':['MAE','MSE','R2'],'Values ':[LinearReg_MAE,LinearReg_MSE,LinearReg_R2]})
print(report.head())