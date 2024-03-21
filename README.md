# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipment Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in Python for Gradient Design.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given data.

## Program:
'''
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: 
RegisterNumber:  
'''
import numpy as np
import pandas as pd
dataset=pd.read_csv('/content/student_scores.csv')

X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
print(X)
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

Y_pred=reg.predict(X_test)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

plt.scatter(X_train,Y_train,color='green')
plt.plot(X_train,reg.predict(X_train),color='red')
plt.title('training set(h vs s)')
plt.xlabel('hours')
plt.ylabel('scores')

plt.scatter(X_test,Y_test,color='red')
plt.plot(X_test,reg.predict(X_test),color='green')
plt.title('test set(h vs s)')
plt.xlabel('hours')
plt.ylabel('scores')

mse=mean_squared_error(Y_test,Y_pred)
print('MSC=',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE=',mae)

rmse=np.sqrt(mse)
print("RMSE=",rmse)
## Output:
![image](https://github.com/logesh1326/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/153622874/45638176-40c6-4ac3-bf84-f502393c2811)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.


https://github.com/logesh1326/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/main/README.md
