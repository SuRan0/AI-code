from sklearn import tree

# prepare training data
# X is input data, row: sample, column: feature
X = [[0, 0], [1, 1]]
y = [0, 1]

# create and train model
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

# predict
print(clf.predict([[2., 2.]]))
