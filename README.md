# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SASIDEVI V
RegisterNumber: 212222230136 
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("/content/student_scores (1).csv")
df.head()

df.tail()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

y_pred
y_test
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

## Output:
### df.head()
![ml21](https://user-images.githubusercontent.com/118707332/229969599-4a9566bf-5f36-4449-980b-bbef38f2f67e.png)
### df.tail()
![ml22](https://user-images.githubusercontent.com/118707332/229969618-0c9a22ec-eeff-4ee4-b779-9e7491cc0227.png)
### x
![ml23](https://user-images.githubusercontent.com/118707332/229969632-951da810-37ad-4dd1-b48d-5bae902f390f.png)
### y
![ml24](https://user-images.githubusercontent.com/118707332/229969647-d46f574a-d821-4399-b94d-8515156f2b4e.png)
### y_pred
![ml25](https://user-images.githubusercontent.com/118707332/229969665-af62e371-65ae-4371-bea2-0b578d5dfa11.png)
### Training set
![ml26](https://user-images.githubusercontent.com/118707332/229969682-a94dd702-f59c-44b6-bae6-820a4d6aef0c.png)
### Testing set
![ml27](https://user-images.githubusercontent.com/118707332/229969692-cf549e43-b9f5-498e-b1ef-0cfe1fbd92a4.png)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
