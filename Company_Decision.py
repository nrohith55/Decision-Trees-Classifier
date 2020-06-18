# -*- coding: utf-8 -*-
"""
Created on Thu May 14 01:54:21 2020

@author: User
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


df=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\Decision Trees\\Company_Data.csv")

df["Sales"]=df['Sales'].astype('category')


df['Sales'] = pd.Categorical(df.Sales)
print (df.dtypes)

df.loc[df.Sales<= 10, "Sales"]='No'
df.loc[df.Sales !='No', "Sales"]= 'Yes'

df

df=pd.get_dummies(df,columns=['ShelveLoc','Urban','US'],drop_first=True)
help(pd.get_dummies)

#Model building

train,test=train_test_split(df,test_size=0.20)

model=DecisionTreeClassifier(criterion='entropy')

model.fit(train.iloc[:,1:12],train.iloc[:,0])

#Train accuracy

train_acc=np.mean(model.predict(train.iloc[:,1:12])==train.iloc[:,0])

#Test accuracy

test_acc=np.mean(model.predict(test.iloc[:,1:12])==test.iloc[:,0])

###########################################################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


df=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\Decision Trees\\Company_Data.csv")

df.loc[df.Sales<= 10,"Sales"] = "No"
df.loc[df.Sales!= 'No' ,"Sales"]='Yes'

df=pd.get_dummies(df,columns=['ShelveLoc','Urban','US'],drop_first=True)

X=df.iloc[:,1:12]
y=df.iloc[:,0]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

model=DecisionTreeClassifier(criterion='entropy')
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

from sklearn.metrics import accuracy_score ,confusion_matrix,classification_report

accuracy_score(y_test,y_pred)#0.78
confusion_matrix(y_test,y_pred)

classification_report(y_test,y_pred)
























