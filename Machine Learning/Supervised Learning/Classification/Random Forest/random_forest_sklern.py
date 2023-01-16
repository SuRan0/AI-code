from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pandas as pd

# use scikit-learn dataset
# n_informative: used features, n_redundant: can be descipted by other features
# random_state: random seed, in order to generate the same data for each run
X, y = make_classification(n_features=4, n_informative=2, n_redundant=0, random_state=0) # X: shape(100, 4), y: shape(100, )

# n_estimater: number of decision tree
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)

# clf.predict()

# use self dataset
data = pd.read_csv("my_data.csv")
# delete "label" column, X is feature matrix
X = data.drop("label", axis=1)
y = data["label"]
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X, y)
