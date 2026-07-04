import sklearn as sk
import numpy as np
import pandas as pd
ds=pd.read_csv("/content/Titanic Dataset.csv")
ds
ds.describe()
ds.isnull().sum()
ds["Sex"].fillna(ds["Sex"].mode()[0],inplace=True)
ds["Age"].fillna(ds["Age"].median(),inplace=True)
ds["SibSp"].fillna(0,inplace=True)
ds["Fare"].fillna(7.75,inplace=True)
ds['Cabin'].fillna(ds['Cabin'].mode()[0],inplace=True)
ds["Embarked"].fillna("S",inplace=True)
ds.loc[ds["Name"].isna() & (ds["Sex"]=="male"),"Name"]="MR"
ds.loc[ds["Name"].isnull() & (ds["Sex"]=="female"),"Name"]="MRs"
ds["Sex"]=ds["Sex"].map({"male":1,"female":0})
ds["Embarked"]=ds["Embarked"].map({"Q":0,"C":1,"S":2})
ds.drop(["PassengerId","Pclass","Name","Age","SibSp","Parch","Cabin","Fare","Ticket"],axis=1,inplace=True)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score
X=ds.drop("Survived",axis=1)
y=ds["Survived"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=92)
m=LogisticRegression(max_iter=700)
m.fit(X_train,y_train)
m_p=m.predict(X_test)

print(f"Accuracy is : {accuracy_score(y_test,m_p)}")
print(f"Precision is : {precision_score(y_test,m_p)}")
