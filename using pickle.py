import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score
import pickle
data=pd.read_csv("age_of_marriage_data.csv")
#print(data.head())
#print(data.isnull().sum())
#print(data.shape)
data.dropna(inplace=True)
#print(data.shape)
X=data.loc[:,['gender', 'height', 'religion', 'caste', 'mother_tongue','country']]
y=data.age_of_marriage
#print(X.head())
lb=LabelEncoder()
X.loc[:,['gender', 'religion', 'caste', 'mother_tongue','country']]=data.loc[:,['gender',  'religion', 'caste', 'mother_tongue','country']].apply(lb.fit_transform)
def h_cms(h):
    return(int(h.split('\'')[0])*30.48)+(int(h.split('\'')[1].replace('"',''))*2.54)
X['height_cms']=X.height.apply(h_cms) 

X.drop(['height'],axis=1,inplace=True)
print(X.head())     
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2 ,random_state=0)
model=RandomForestRegressor(n_estimators=100,max_depth=11)
model.fit(X,y)
print(model.score(X_test,y_test))
y_predict= model.predict(X_test)
print(mean_absolute_error(y_test, y_predict))
pickle.dump(model,open("marr_pred.pkl",'wb'))
pickle.load(open("marr_pred.pkl",'rb'))
print(model.predict([[1,2,5,6,5,175]]))

                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                           
# -*- coding: utf-8 -*-

