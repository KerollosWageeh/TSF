import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

src = 'http://bit.ly/w-data'
data = pd.read_csv(src)
hours = data.iloc[:, :-1]
scores = data.iloc[:, 1]

regressor = LinearRegression()
regressor.fit(hours, scores)

line = regressor.coef_ * hours + regressor.intercept_
plt.scatter(hours, scores)
plt.plot(hours, line, color='red')
plt.show()

pred = regressor.predict([[9.25]])
print("No of Hours = {}".format(9.25))
print("Predicted Score = {}".format(pred[0]))
