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
Knn=KNeighborsClassifier(n_neighbors=4).fit(x_train,y_train)
predictions=Knn.predict(x_test)
knn_accuracy_score=metrics.accuracy_score(y_test,predictions)
knn_jaccardIndex=metrics.jaccard_score(y_test,predictions)
knn_F1Score=metrics.f1_score(y_test,predictions)
knn_report=pd.DataFrame({'Metrics ':['Acc_ScoreKNN','JaccInded_KNN','F1_Score'],'Values ':[knn_accuracy_score,knn_jaccardIndex,knn_F1Score]})
final_report=pd.concat([report,knn_report],ignore_index=True)
print(final_report)
Tree=DecisionTreeClassifier()
Tree.fit(x_train,y_train)
prediction=Tree.predict(x_test)
tree_accuracy_score=metrics.accuracy_score(y_test,prediction)
tree_jaccard_index=metrics.jaccard_score(y_test,prediction)
tree_f1_score=metrics.f1_score(y_test,prediction)
tree_report=pd.DataFrame({'Metrics ':['Tree_Acc','Tree_jaccIndex','Tree_F1'],'Values ':[tree_accuracy_score,tree_jaccard_index,tree_f1_score]})
final_report=pd.concat([final_report,tree_report],ignore_index=True)
print(final_report)
Lr=LogisticRegression(solver='liblinear')
Lr.fit(x_train,y_train)
prediction=Lr.predict(x_test)
prediction_prob=Lr.predict_proba(x_test)
Lr_Accuracy_score=metrics.accuracy_score(y_test,prediction)
Lr_jaccard_index=metrics.jaccard_score(y_test,prediction)
Lr_f1_score=metrics.f1_score(y_test,prediction)
Lr_log_loss=metrics.log_loss(y_test,prediction)
log_report=pd.DataFrame({'Metrics ':['log_acc','logg_jacc','log_f1','log_loss'],'Values ':[Lr_Accuracy_score,Lr_jaccard_index,Lr_f1_score,Lr_log_loss]})
final_report=pd.concat([final_report,log_report],ignore_index=True)
print(final_report)
Svm=svm.SVC(kernel='linear')
Svm.fit(x_train,y_train)
prediction=Svm.predict(x_test)
svm_acc=metrics.accuracy_score(y_test,prediction)
svm_jacc=metrics.jaccard_score(y_test,prediction)
svm_f1_score=metrics.f1_score(y_test,prediction)
svm_report=pd.DataFrame({'Metrics ':['svm_acc','svm_jacc','svm_f1'],'Values ':[svm_acc,svm_jacc,svm_f1_score]})
final_report=pd.concat([final_report,svm_report],ignore_index=True)
print(final_report)
