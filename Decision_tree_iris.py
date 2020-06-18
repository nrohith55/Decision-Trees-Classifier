# -*- coding: utf-8 -*-
"""
Created on Thu May 14 01:06:02 2020

@author: Rohith
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#Importing data set
df=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\Decision Trees\\iris.csv")
#Devide the data set into train and test
train,test=train_test_split(df,test_size=0.2)

#Model building
help(DecisionTreeClassifier)
model=DecisionTreeClassifier(criterion='entropy')
model.fit(train.iloc[:,0:4],train.iloc[:,4])

train_acc=np.mean(model.predict(train.iloc[:,0:4])==train.iloc[:,4])
test_acc=np.mean(model.predict(test.iloc[:,0:4])==test.iloc[:,4])


#########################Type 2#################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


df=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\Decision Trees\\iris.csv")

X=df.iloc[:,0:4]
y=df.iloc[:,4]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

 model=DecisionTreeClassifier(criterion='entropy')

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report
confusion_matrix(y_test,y_pred)
classification_report(y_test,y_pred)

##################################################################################################################





























