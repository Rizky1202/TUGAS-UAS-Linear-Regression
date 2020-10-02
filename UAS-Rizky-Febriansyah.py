# Simple Linear Regression

# Import Library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

# Memanggil dataset

datasets = pd.read_csv('Salary.csv')

#Sumbu X adalah Pengalaman Kerja dan Sumbu Y adalah Gaji
X = datasets.iloc[:, :-1].values
Y = datasets.iloc[:, 1].values

# Melakukan Splitting dataset pada Training set dan Test set

from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

# Melakukan Fitting Simple Linear Regression pada training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_Train, Y_Train)

# Prediksi hasil Test set ￼

Y_Pred = regressor.predict(X_Test)


# Visualisisasi hasil Training

plt.scatter(X_Train, Y_Train, color = 'red')
plt.plot(X_Train, regressor.predict(X_Train), color = 'blue')
plt.title('Gaji vs Pengalaman (Training Set)')
plt.xlabel('Pengalaman Kerja')
plt.ylabel('Gaji')
plt.show()

# Visualisisasi hasil Test set

plt.scatter(X_Test, Y_Test, color = 'red')
plt.plot(X_Train, regressor.predict(X_Train), color = 'blue')
plt.title('Gaji vs Pengalaman (Training Set)')
plt.xlabel('Pengalaman Kerja')
plt.ylabel('Gaji')
plt.show()