# KEROLLOS WAGEEH

import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.linear_model import LinearRegression

# Reading data form remote link and converting it to DataFrame
src = 'http://bit.ly/w-data'
data = pd.read_csv(src)

# Dividing the data into hours and scores
hours = data.iloc[:, :-1]
scores = data.iloc[:, 1]

# Training the Model
regressor = LinearRegression()
regressor.fit(hours, scores)

# Plotting the data and the regression line
line = regressor.coef_ * hours + regressor.intercept_
plt.scatter(hours, scores)
plt.plot(hours, line, color='red')
plt.show()

# Predict the score when study hours = 9.25
pred = regressor.predict([[9.25]])
print("No of Hours = {}".format(9.25))
print("Predicted Score = {}".format(pred[0]))
