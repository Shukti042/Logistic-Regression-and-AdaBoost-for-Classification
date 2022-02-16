from numpy.random.mtrand import RandomState
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
from numpy.random import choice
import scipy.stats as stats
import sys
f = open("out.txt", "w")

def predict(x,w):
  hwx=np.tanh(np.dot(x,w))
  y_hat=np.array([1 if x>=0 else -1 for x in hwx]).reshape(hwx.shape[0],1).tolist()
  return y_hat
def adaboost_predict(hypothesis,hypothesis_weight,x,y):
    h=np.zeros((y.shape[0],1))
    for index,k in enumerate(hypothesis_weight):
        hwx=np.tanh(np.dot(x,hypothesis[index]))
        # y_hat=np.array([1 if x>=0 else -1 for x in hwx]).reshape(hwx.shape[0],1)
        h=h+(k*hwx)
    # h=h/sum(hypothesis_weight)
    y_hat=np.array([1 if x>=0 else -1 for x in h]).reshape(h.shape[0],1).tolist()
    compute_metric(y,y_hat)

def adaboost(traindf,test_x,test_y,k=5,alpha=0.00005,trainingsteps=1000):
  weight=np.full((traindf.shape[0]),1/traindf.shape[0]).tolist()
  hypothesis=[]
  hypothesis_weight=[]
  for i in range(k):
    tempdf=traindf.sample(n=traindf.shape[0],replace=True, weights=weight,random_state=7)
    train_y=tempdf.pop((traindf.shape[1]-1)).to_numpy()
    train_y=train_y.reshape((train_y.shape[0],1))
    train_x=tempdf.to_numpy()
    w=train(train_x,train_y,alpha=alpha,trainingsteps=trainingsteps)
    hypothesis.append(w)
    hwx=np.tanh(np.dot(test_x,w))
    y_hat=np.array([1 if x>=0 else -1 for x in hwx]).reshape(hwx.shape[0],1).tolist()
    y=test_y.tolist()
    err=0
    for index,item in enumerate(y):
      if item[0]!=y_hat[index][0]:
        err=err+weight[index]
    if err>0.5:
      continue
    for index,item in enumerate(y):
      if item[0]==y_hat[index][0]:
        weight[index]=(weight[index]*err)/(1-err)
    weightsum=sum(weight)
    for i in range(len(weight)):
      weight[i]=weight[i]/weightsum
    z=math.log((1-err)/err)
    hypothesis_weight.append(z)
  return hypothesis,hypothesis_weight

def compute_metric(y,y_hat):
    TP=0;TN=0;FP=0;FN=0
    for index,item in enumerate(y):
        if item[0]==-1 and y_hat[index][0]==-1:
            TN+=1
        if item[0]==1 and y_hat[index][0]==1:
            TP+=1
        if item[0]==1 and y_hat[index][0]==-1:
            FN+=1
        if item[0]==-1 and y_hat[index][0]==1:
            FP+=1
    f.write("Accuracy : "+str((TP+TN)/(TP+TN+FP+FN))+"\n")
    if TP+FN!=0:
        f.write("Sensitivity : "+str(TP/(TP+FN))+"\n")
    else:
        f.write("Sensitivity : 1\n")
    if TN+FP!=0:
        f.write("Specificity : "+str(TN/(TN+FP))+"\n")
    else:
       f.write("Specificity : 1\n")
    if TP+FP!=0:
        f.write("Precision : "+str(TP/(TP+FP))+"\n")
    else:
        f.write("Precision : 1\n")
    if TP+FP!=0:
        f.write("False discovery rate : "+str(FP/(TP+FP))+"\n")
    else:
        f.write("False discovery rate : 1\n")
    if TP+FP+FN!=0:
       f.write("F1 : "+str((2*TP)/(2*TP+FP+FN))+"\n")
    else:
        f.write("F1 : 1\n")
def train(train_x,train_y,alpha=0.0005,earlystop=0.5,trainingsteps=10000):
  np.random.seed(7)
  # w=np.random.rand(train_x.shape[1],1)
  w=np.zeros((train_x.shape[1],1))
  w[0][0]=0
  # train_x,validation_x,train_y,validation_y=train_test_split(train_x,train_y,test_size=0.2, random_state=42, shuffle=True)
  # x=[]
  # y=[]
  # yvalid=[]
  # initalpha=alpha
  for i in range(trainingsteps):
    hwx=np.tanh(np.dot(train_x,w))
    w=w+alpha*np.dot(train_x.T,((train_y-hwx)*(1-np.square(hwx))))
    # x.append(i)
    loss=np.sum(np.power((train_y-hwx),2))/train_x.shape[0]
    # y.append(loss)
    # hvalid=np.tanh(np.dot(validation_x,w))
    # valid_loss=np.sum(np.power((validation_y-hvalid),2))/hvalid.shape[0]
    # yvalid.append(valid_loss)
    if loss<earlystop:
      break
    
  # plt.plot(x,y,label = "train loss")
  # plt.plot(x,yvalid,label = "validation loss")
  # plt.legend()
  # plt.show()
  # hwx=np.tanh(np.dot(train_x,w))
  # loss=np.sum(np.power((train_y-hwx),2))/hwx.shape[0]
  # print("\nTrain Set Loss: ",loss)
  # hwx=np.tanh(np.dot(validation_x,w))
  # loss=np.sum(np.power((validation_y-hwx),2))/hwx.shape[0]
  # print("Validation Set Loss: ",loss)
  # print("\nTrain Set : ")
  # hwx=np.tanh(np.dot(train_x,w))
  # y_hat=np.array([1 if x>=0 else -1 for x in hwx]).reshape(hwx.shape[0],1).tolist()
  # compute_metric(train_y.tolist(),y_hat)
  # print("\nValidation Set :")
  # hwx=np.tanh(np.dot(validation_x,w))
  # y_hat=np.array([1 if x>=0 else -1 for x in hwx]).reshape(hwx.shape[0],1).tolist()
  # compute_metric(validation_y.tolist(),y_hat)
  return w

def B(q):
  return -1*(q*math.log(q,2)+(1-q)*math.log(1-q,2))

def continuous_gain(df,column,target_column):
  total_data_count=np.sum(df[target_column].value_counts())
  df_column={}
  df_column["a"]=df[(df[column] >=df[column].min()) & (df[column] <df[column].median())]
  df_column["b"]=df[(df[column] >=df[column].median()) & (df[column] <=df[column].max())]
  totalentropy=0
  for key, value in df_column.items():
    pn=value[target_column].value_counts()
    entropy=0
    if len(pn)<2:
      entropy=0
    else:
      entropy=B(pn[1]/np.sum(pn))
    entropy=entropy*(np.sum(pn)/total_data_count)
    totalentropy+=entropy
  return totalentropy
def category_gain(df,column,target_column):
  total_data_count=np.sum(df[target_column].value_counts())
  df_column={}
  column_values=df[column].unique()
  totalentropy=0
  for c in column_values:
    df_column[c]=df[df[column] == c]
  for key, value in df_column.items():
    pn=value[target_column].value_counts()
    entropy=0
    if len(pn)<2:
      entropy=0
    else:
      entropy=B(pn[1]/np.sum(pn))
    entropy=entropy*(np.sum(pn)/total_data_count)
    totalentropy+=entropy
  return totalentropy
def calculate_information_gain1(df):
  # print(list(df.columns))
  pn=df['Churn'].value_counts()
  total_data_count=np.sum(pn)
  information_gain={}
  base_entropy=B(pn[1]/total_data_count)
  categorical_column=['gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingMovies','StreamingTV','Contract','PaperlessBilling','PaymentMethod']
  for item in categorical_column:
    entropy=category_gain(df,item,'Churn')
    information_gain[item]=base_entropy-entropy


  # print(df['MonthlyCharges'].min(),df['MonthlyCharges'].max() )
  df_MonthlyCharges={}
  df_MonthlyCharges["a"]=df[(df['MonthlyCharges'] >18) & (df['MonthlyCharges'] <40)]
  df_MonthlyCharges["b"]=df[(df['MonthlyCharges'] >=40) & (df['MonthlyCharges'] <60)]
  df_MonthlyCharges["c"]=df[(df['MonthlyCharges'] >=60) & (df['MonthlyCharges'] <80)]
  df_MonthlyCharges["d"]=df[(df['MonthlyCharges'] >=80) & (df['MonthlyCharges'] <100)]
  df_MonthlyCharges["e"]=df[(df['MonthlyCharges'] >=100) & (df['MonthlyCharges'] <120)]
  information_gain['MonthlyCharges']=base_entropy
  for key, value in df_MonthlyCharges.items():
    # filename=key+".csv"
    # value.to_csv(filename)
    pn=value['Churn'].value_counts()
    entropy=B(pn[1]/np.sum(pn))
    entropy=entropy*(np.sum(pn)/total_data_count)
    information_gain['MonthlyCharges']=information_gain['MonthlyCharges']-entropy
  

  # print(df['TotalCharges'].min(),df['TotalCharges'].max())
  df_TotalCharges={}
  df_TotalCharges["a"]=df[(df['TotalCharges'] >18) & (df['TotalCharges'] <1000)]
  df_TotalCharges["b"]=df[(df['TotalCharges'] >=1000) & (df['TotalCharges'] <2000)]
  df_TotalCharges["c"]=df[(df['TotalCharges'] >=2000) & (df['TotalCharges'] <3000)]
  df_TotalCharges["d"]=df[(df['TotalCharges'] >=3000) & (df['TotalCharges'] <4000)]
  df_TotalCharges["e"]=df[(df['TotalCharges'] >=4000) & (df['TotalCharges'] <5000)]
  df_TotalCharges["f"]=df[(df['TotalCharges'] >=5000) & (df['TotalCharges'] <6000)]
  df_TotalCharges["g"]=df[(df['TotalCharges'] >=6000) & (df['TotalCharges'] <7000)]
  df_TotalCharges["h"]=df[(df['TotalCharges'] >=7000) & (df['TotalCharges'] <8000)]
  df_TotalCharges["i"]=df[(df['TotalCharges'] >=8000) & (df['TotalCharges'] <9000)]
  information_gain['TotalCharges']=base_entropy
  for key, value in df_TotalCharges.items():
    # filename=key+".csv"
    # value.to_csv(filename)
    pn=value['Churn'].value_counts()
    entropy=B(pn[1]/np.sum(pn))
    entropy=entropy*(np.sum(pn)/total_data_count)
    information_gain['TotalCharges']=information_gain['TotalCharges']-entropy

  # print(df['tenure'].min(),df['tenure'].max())
  df_tenure={}
  df_tenure["a"]=df[(df['tenure'] >0) & (df['tenure'] <10)]
  df_tenure["b"]=df[(df['tenure'] >=10) & (df['tenure'] <20)]
  df_tenure["c"]=df[(df['tenure'] >=20) & (df['tenure'] <30)]
  df_tenure["d"]=df[(df['tenure'] >=30) & (df['tenure'] <40)]
  df_tenure["e"]=df[(df['tenure'] >=40) & (df['tenure'] <50)]
  df_tenure["f"]=df[(df['tenure'] >=50) & (df['tenure'] <60)]
  df_tenure["g"]=df[(df['tenure'] >=60) & (df['tenure'] <70)]
  df_tenure["h"]=df[(df['tenure'] >=70) & (df['tenure'] <80)]
  information_gain['tenure']=base_entropy
  for key, value in df_tenure.items():
    # filename=key+".csv"
    # value.to_csv(filename)
    pn=value['Churn'].value_counts()
    entropy=B(pn[1]/np.sum(pn))
    entropy=entropy*(np.sum(pn)/total_data_count)
    information_gain['tenure']=information_gain['tenure']-entropy

  sort_orders = sorted(information_gain.items(), key=lambda x: x[1], reverse=True)
  # print(sort_orders)
  sorted_features=[i[0] for i in sort_orders]
  return sorted_features[:8]

def calculate_information_gain2(df):
  # print(list(df.columns))
  pn=df['income'].value_counts()
  total_data_count=np.sum(pn)
  information_gain={}
  base_entropy=B(pn[1]/total_data_count)
  categorical_column=['workclass','education','maritalstatus','occupation','relationship','race','sex','nativecountry']
  for item in categorical_column:
    entropy=category_gain(df,item,'income')
    information_gain[item]=base_entropy-entropy
  
  # print(df['age'].min(),df['age'].max())
  df_age={}
  df_age["a"]=df[(df['age'] >15) & (df['age'] <30)]
  df_age["b"]=df[(df['age'] >=30) & (df['age'] <50)]
  df_age["c"]=df[(df['age'] >=50) & (df['age'] <60)]
  df_age["d"]=df[(df['age'] >=60) & (df['age'] <70)]
  df_age["e"]=df[(df['age'] >=70) & (df['age'] <100)]
  information_gain['age']=base_entropy
  for key, value in df_age.items():
    # filename=key+".csv"
    # value.to_csv(filename)
    pn=value['income'].value_counts()
    entropy=0
    if len(pn)<2:
      entropy=0
    else:
      entropy=B(pn[1]/np.sum(pn))
    entropy=entropy*(np.sum(pn)/total_data_count)
    information_gain['age']=information_gain['age']-entropy

  # print(df['fnlwgt'].min(),df['fnlwgt'].max())
  df_fnlwgt={}
  df_fnlwgt["a"]=df[(df['fnlwgt'] >12000) & (df['fnlwgt'] <300000)]
  df_fnlwgt["b"]=df[(df['fnlwgt'] >=300000) & (df['fnlwgt'] <600000)]
  df_fnlwgt["c"]=df[(df['fnlwgt'] >=600000) & (df['fnlwgt'] <900000)]
  df_fnlwgt["d"]=df[(df['fnlwgt'] >=900000) & (df['fnlwgt'] <1200000)]
  df_fnlwgt["e"]=df[(df['fnlwgt'] >=1200000) & (df['fnlwgt'] <1500000)]
  information_gain['fnlwgt']=base_entropy
  for key, value in df_fnlwgt.items():
    # filename=key+".csv"
    # value.to_csv(filename)
    pn=value['income'].value_counts()
    entropy=0
    if len(pn)<2:
      entropy=0
    else:
      entropy=B(pn[1]/np.sum(pn))
    entropy=entropy*(np.sum(pn)/total_data_count)
    information_gain['fnlwgt']=information_gain['fnlwgt']-entropy
  
  # print(df['educationnum'].min(),df['educationnum'].max())
  df_educationnum={}
  df_educationnum["a"]=df[(df['educationnum'] >0) & (df['educationnum'] <4)]
  df_educationnum["b"]=df[(df['educationnum'] >=4) & (df['educationnum'] <8)]
  df_educationnum["c"]=df[(df['educationnum'] >=8) & (df['educationnum'] <12)]
  df_educationnum["d"]=df[(df['educationnum'] >=12) & (df['educationnum'] <=16)]
  information_gain['educationnum']=base_entropy
  for key, value in df_educationnum.items():
    # filename=key+".csv"
    # value.to_csv(filename)
    pn=value['income'].value_counts()
    entropy=0
    if len(pn)<2:
      entropy=0
    else:
      entropy=B(pn[1]/np.sum(pn))
    entropy=entropy*(np.sum(pn)/total_data_count)
    information_gain['educationnum']=information_gain['educationnum']-entropy

  # print(df['capitalgain'].min(),df['capitalgain'].max())
  df_capitalgain={}
  df_capitalgain["a"]=df[(df['capitalgain'] >=0) & (df['capitalgain'] <20000)]
  df_capitalgain["b"]=df[(df['capitalgain'] >=20000) & (df['capitalgain'] <40000)]
  df_capitalgain["c"]=df[(df['capitalgain'] >=40000) & (df['capitalgain'] <60000)]
  df_capitalgain["d"]=df[(df['capitalgain'] >=60000) & (df['capitalgain'] <80000)]
  df_capitalgain["e"]=df[(df['capitalgain'] >=80000) & (df['capitalgain'] <100000)]
  information_gain['capitalgain']=base_entropy
  for key, value in df_capitalgain.items():
    # filename=key+".csv"
    # value.to_csv(filename)
    pn=value['income'].value_counts()
    entropy=0
    if len(pn)<2:
      entropy=0
    else:
      entropy=B(pn[1]/np.sum(pn))
    entropy=entropy*(np.sum(pn)/total_data_count)
    information_gain['capitalgain']=information_gain['capitalgain']-entropy
  # print(df['capitalloss'].min(),df['capitalloss'].max())
  df_capitalloss={}
  df_capitalloss["a"]=df[(df['capitalloss'] >=0) & (df['capitalloss'] <1000)]
  df_capitalloss["b"]=df[(df['capitalloss'] >=1000) & (df['capitalloss'] <2000)]
  df_capitalloss["c"]=df[(df['capitalloss'] >=2000) & (df['capitalloss'] <3000)]
  df_capitalloss["d"]=df[(df['capitalloss'] >=3000) & (df['capitalloss'] <4000)]
  df_capitalloss["e"]=df[(df['capitalloss'] >=4000) & (df['capitalloss'] <5000)]
  information_gain['capitalloss']=base_entropy
  for key, value in df_capitalloss.items():
    # filename=key+".csv"
    # value.to_csv(filename)
    pn=value['income'].value_counts()
    entropy=0
    if len(pn)<2:
      entropy=0
    else:
      entropy=B(pn[1]/np.sum(pn))
    entropy=entropy*(np.sum(pn)/total_data_count)
    information_gain['capitalloss']=information_gain['capitalloss']-entropy

  # print(df['hoursperweek'].min(),df['hoursperweek'].max())
  df_hoursperweek={}
  df_hoursperweek["a"]=df[(df['hoursperweek'] >0) & (df['hoursperweek'] <20)]
  df_hoursperweek["b"]=df[(df['hoursperweek'] >=20) & (df['hoursperweek'] <40)]
  df_hoursperweek["c"]=df[(df['hoursperweek'] >=40) & (df['hoursperweek'] <60)]
  df_hoursperweek["d"]=df[(df['hoursperweek'] >=60) & (df['hoursperweek'] <80)]
  df_hoursperweek["e"]=df[(df['hoursperweek'] >=80) & (df['hoursperweek'] <100)]
  information_gain['hoursperweek']=base_entropy
  for key, value in df_hoursperweek.items():
    # filename=key+".csv"
    # value.to_csv(filename)
    pn=value['income'].value_counts()
    entropy=0
    if len(pn)<2:
      entropy=0
    else:
      entropy=B(pn[1]/np.sum(pn))
    entropy=entropy*(np.sum(pn)/total_data_count)
    information_gain['hoursperweek']=information_gain['hoursperweek']-entropy
  sort_orders = sorted(information_gain.items(), key=lambda x: x[1], reverse=True)
  # # print(sort_orders)
  sorted_features=[i[0] for i in sort_orders]
  return sorted_features[:10]

def calculate_information_gain3(df):
  # print(list(df.columns))
  pn=df['Class'].value_counts()
  total_data_count=np.sum(pn)
  information_gain={}
  base_entropy=B(pn[1]/total_data_count)
  continuous_columns=df.columns.tolist()
  continuous_columns.remove('Class')
  for item in continuous_columns:
    entropy=continuous_gain(df,item,'Class')
    information_gain[item]=base_entropy-entropy
  sort_orders = sorted(information_gain.items(), key=lambda x: x[1], reverse=True)
  #print(sort_orders)
  sorted_features=[i[0] for i in sort_orders]
  return sorted_features[:10]

def preprocess1(datasource):
  df = pd.read_csv(datasource)
  df.dropna(inplace = True)
  df.pop("customerID")
  # print(df['gender'].value_counts().idxmax())
  for i, j in df.iterrows():
    for index,item in enumerate(j):
      if str.strip(str(item))=='':
        df.at[i,df.columns[index]]=None
  # print(df.isnull().sum())
  # print(df.dtypes)
  for index,item in enumerate(df['TotalCharges']):
    if df.at[index,'TotalCharges']:
      df.at[index,'TotalCharges']=float(df.at[index,'TotalCharges'])
  df['TotalCharges'].fillna(df['TotalCharges'].mean(),inplace = True)
  # print(df.shape)
  preferred_columns=calculate_information_gain1(df)
  preferred_columns.append("Churn")
  df=df[preferred_columns]
  # print(df.dtypes) 
  # print(df['tenure'].mean())
  column_to_be_encoded=['Contract', 'OnlineSecurity', 'TechSupport', 'InternetService', 'OnlineBackup', 'PaymentMethod', 'DeviceProtection']
  newdf=pd.get_dummies(df,columns=column_to_be_encoded)
  newdf['Churn'].replace('No',-1,inplace=True)
  newdf['Churn'].replace('Yes',1,inplace=True)
  # print(newdf.to_string())
  # newdf.to_csv('out.csv')
  scaler = preprocessing.StandardScaler()
  newdf[['tenure']]=scaler.fit_transform(newdf[['tenure']])
  z_scores = stats.zscore(newdf)
  abs_z_scores = np.abs(z_scores)
  filtered_entries = (abs_z_scores < 3).all(axis=1)
  newdf = newdf[filtered_entries]
  # newdf.to_csv('out.csv')
  # newdf.to_csv('out.csv')
  Y=newdf.pop("Churn")
  train_x,test_x,train_y,test_y=train_test_split(newdf,Y,test_size=0.2, random_state=1, shuffle=True)
  train_x=np.hstack((np.ones((train_x.shape[0],1)),train_x))
  test_x=np.hstack((np.ones((test_x.shape[0],1)),test_x))
  train_y=train_y.to_numpy()
  train_y=train_y.reshape((train_y.shape[0],1))
  test_y=test_y.to_numpy()
  test_y=test_y.reshape((test_y.shape[0],1))
  return train_x,train_y,test_x,test_y
def preprocess2(traindatasource,testdatasource):
  columns=['age','workclass','fnlwgt','education','educationnum','maritalstatus','occupation','relationship','race','sex','capitalgain','capitalloss','hoursperweek','nativecountry','income']
  traindf = pd.read_csv(traindatasource,header=None,sep=', ',engine='python',names=columns)
  for i, j in traindf.iterrows():
    for index,item in enumerate(j):
      if str.strip(str(item))=='?':
        traindf.at[i,traindf.columns[index]]=traindf[traindf.columns[index]].mode()[0]
  testdf = pd.read_csv(testdatasource,header=None,sep=', ',engine='python',names=columns)
  testdf = testdf.iloc[1: , :]
  testdf['income'].replace(regex=True, to_replace=r'\.', value='', inplace=True)
  for i, j in testdf.iterrows():
    for index,item in enumerate(j):
      if str.strip(str(item))=='?':
        testdf.at[i,testdf.columns[index]]=testdf[testdf.columns[index]].mode()[0]
  testdf['age']=testdf['age'].astype('int64')
  df=pd.concat([traindf, testdf])
  preferred_columns=calculate_information_gain2(df)
  # print(preferred_columns)
  preferred_columns.append("income")
  traindf=traindf[preferred_columns]
  testdf=testdf[preferred_columns]
  # # print(df.dtypes) 
  # # print(df['tenure'].mean())
  scaler = preprocessing.StandardScaler()
  column_to_be_encoded=['relationship', 'maritalstatus', 'education','occupation','sex','workclass']
  newtraindf=pd.get_dummies(traindf,columns=column_to_be_encoded)
  newtraindf['income'].replace('<=50K',-1,inplace=True)
  newtraindf['income'].replace('>50K',1,inplace=True)
  newtraindf[['educationnum','age','hoursperweek','capitalgain']]=scaler.fit_transform(newtraindf[['educationnum','age','hoursperweek','capitalgain']])
  newtestdf=pd.get_dummies(testdf,columns=column_to_be_encoded)
  newtestdf['income'].replace('<=50K',-1,inplace=True)
  newtestdf['income'].replace('>50K',1,inplace=True)
  newtestdf[['educationnum','age','hoursperweek','capitalgain']]=scaler.fit_transform(newtestdf[['educationnum','age','hoursperweek','capitalgain']])
  # print(np.sum(newtestdf['income']))
  # print(np.sum(newtraindf['income']))
  train_y=newtraindf.pop("income").to_numpy()
  train_y=train_y.reshape((train_y.shape[0],1))
  train_x=newtraindf.to_numpy()
  test_y=newtestdf.pop("income").to_numpy()
  test_y=test_y.reshape((test_y.shape[0],1))
  test_x=newtestdf.to_numpy()
  return train_x,train_y,test_x,test_y
def preprocess3(datasource):
  initialdf = pd.read_csv(datasource)
  initialdf.dropna(inplace = True)
  ntosample=initialdf['Class'].value_counts()[1]*50
  pdf=initialdf[initialdf['Class'] == 1]
  ndf=initialdf[initialdf['Class'] == 0]
  ndf=ndf.sample(n=ntosample,replace=False,random_state=7)
  df=pd.concat([ndf, pdf])
  df=df.sample(frac=1,random_state=7).reset_index(drop=True)
  preferred_columns=calculate_information_gain3(df)
  continuous_column=preferred_columns.copy()
  preferred_columns.append("Class")
  df=df[preferred_columns]
  df['Class'].replace(0,-1,inplace=True)
  scaler = preprocessing.StandardScaler()
  df[continuous_column]=scaler.fit_transform(df[continuous_column])
  Y=df.pop("Class")
  train_x,test_x,train_y,test_y=train_test_split(df,Y,test_size=0.2, random_state=42, shuffle=True)
  train_x=np.hstack((np.ones((train_x.shape[0],1)),train_x))
  test_x=np.hstack((np.ones((test_x.shape[0],1)),test_x))
  train_y=train_y.to_numpy()
  train_y=train_y.reshape((train_y.shape[0],1))
  test_y=test_y.to_numpy()
  test_y=test_y.reshape((test_y.shape[0],1))
  return train_x,train_y,test_x,test_y

def main():
    if len(sys.argv)<5:
      print("Please Provide the path to Dataset 1,Train Dataset 2,Test Dataset 2, Dataset 3 Sequentially")
      return
    
    #Experiment on Dataset 1
    f.write("Dataset 1:\n#####################\n")
    train_x,train_y,test_x,test_y=preprocess1(sys.argv[1])
    w=train(train_x,train_y,alpha=0.00005,trainingsteps=10000)
    f.write("\nLogistic Regression :\n---------------------\n")
    f.write("\nTrain Set :\n")
    y_hat=predict(train_x,w)
    compute_metric(train_y.tolist(),y_hat)
    f.write("\nTest Set :\n")
    y_hat=predict(test_x,w)
    compute_metric(test_y.tolist(),y_hat)
    k_list=[5,10,15,20]
    for k in k_list:
      f.write("\n\nAdaboost : Round = "+str(k)+"\n---------------------\n")
      traindf=np.concatenate((train_x,train_y),axis=1)
      traindf = pd.DataFrame(traindf)
      hypothesis,hypothesis_weight=adaboost(traindf,train_x,train_y,k=k)
      f.write("\nTrain Set :\n")
      adaboost_predict(hypothesis,hypothesis_weight,train_x,train_y)
      f.write("\nTest Set :\n")
      adaboost_predict(hypothesis,hypothesis_weight,test_x,test_y)

    #Experiment on Dataset 2
    f.write("\nDataset 2:\n#####################\n")
    train_x,train_y,test_x,test_y=preprocess2(sys.argv[2],sys.argv[3])
    w=train(train_x,train_y,alpha=0.00005,trainingsteps=10000)
    f.write("\nLogistic Regression :\n---------------------\n")
    f.write("\nTrain Set :\n")
    y_hat=predict(train_x,w)
    compute_metric(train_y.tolist(),y_hat)
    f.write("\nTest Set :\n")
    y_hat=predict(test_x,w)
    compute_metric(test_y.tolist(),y_hat)
    k_list=[5,10,15,20]
    for k in k_list:
      f.write("\n\nAdaboost : Round = "+str(k)+"\n---------------------\n")
      traindf=np.concatenate((train_x,train_y),axis=1)
      traindf = pd.DataFrame(traindf)
      hypothesis,hypothesis_weight=adaboost(traindf,train_x,train_y,k=k)
      f.write("\nTrain Set :\n")
      adaboost_predict(hypothesis,hypothesis_weight,train_x,train_y)
      f.write("\nTest Set :\n")
      adaboost_predict(hypothesis,hypothesis_weight,test_x,test_y)

    #Experiment on Dataset 3
    f.write("\nDataset 3:\n#####################\n")
    train_x,train_y,test_x,test_y=preprocess3(sys.argv[4])
    w=train(train_x,train_y,alpha=0.0001,trainingsteps=50000)
    f.write("\nLogistic Regression :\n---------------------\n")
    f.write("\nTrain Set :\n")
    y_hat=predict(train_x,w)
    compute_metric(train_y.tolist(),y_hat)
    f.write("\nTest Set :\n")
    y_hat=predict(test_x,w)
    compute_metric(test_y.tolist(),y_hat)
    k_list=[5,10,15,20]
    for k in k_list:
      f.write("\n\nAdaboost : Round = "+str(k)+"\n---------------------\n")
      traindf=np.concatenate((train_x,train_y),axis=1)
      traindf = pd.DataFrame(traindf)
      hypothesis,hypothesis_weight=adaboost(traindf,train_x,train_y,k=k,alpha=0.0001,trainingsteps=5000)
      f.write("\nTrain Set :\n")
      adaboost_predict(hypothesis,hypothesis_weight,train_x,train_y)
      f.write("\nTest Set :\n")
      adaboost_predict(hypothesis,hypothesis_weight,test_x,test_y)
    
    f.close()

if __name__ == "__main__":
    main()