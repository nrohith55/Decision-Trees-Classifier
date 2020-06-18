# -*- coding: utf-8 -*-
"""
Created on Thu May 14 02:17:41 2020

@author: Rohith
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import scale

df=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\Decision Trees\\Fraud_check.csv")

df_new=df.rename(columns={'Taxable.Income':'Tax','Marital.Status':'Status','City.Population':'Population','Work.Experience':'Exp'})

df_new.loc[df_new.Tax <= 30000, "Tax"]='Risk'
df_new.loc[df_new.Tax !='Risk', "Tax"]= 'Good'
df_new=df_new.iloc[:,[2,0,1,3,4,5]]
df_new=pd.get_dummies(df_new,columns=['Undergrad','Status','Urban'],drop_first=True)

train,test=train_test_split(df_new,test_size=0.2)

#Model building
help(DecisionTreeClassifier)
model=DecisionTreeClassifier(criterion='entropy')
model.fit(train.iloc[:,1:7],train.iloc[:,0])

train_acc=np.mean(model.predict(train.iloc[:,1:7])==train.iloc[:,0])
test_acc=np.mean(model.predict(test.iloc[:,1:7])==test.iloc[:,0])
============================================================================================================================
X=df_new.iloc[:,1:7]
y=df_new.iloc[:,0]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

model=DecisionTreeClassifier(criterion='entropy')

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
classification_report(y_test,y_pred)

from sklearn.tree import plot_tree
plot_tree(model)



########################################################################################################################################














