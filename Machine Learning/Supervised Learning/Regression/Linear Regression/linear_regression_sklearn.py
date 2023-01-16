import numpy as np
from sklearn.linear_model import LinearRegression

# generate training data
x = np.array([[1], [2], [3], [4], [5]])
y = np.array([5, 7, 9, 11, 13])

# create and train the model
reg = LinearRegression().fit(x, y)

# predict
print(reg.predict(np.array([[6]])))

# print coefficient: y=ax+b
print("a: ", reg.coef_)
print("b: ", reg.intercept_)
