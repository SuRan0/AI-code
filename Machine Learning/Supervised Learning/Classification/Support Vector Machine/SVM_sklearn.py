from sklearn.svm import SVC
from sklearn.datasets import make_classification

# use scikit-learn dataset
# n_informative: used features, n_redundant: can be descipted by other features
# random_state: random seed, in order to generate the same data for each run
X, y = make_classification(n_features=4, n_informative=2, n_redundant=0, random_state=0)

# SVC: Support Vector Classification, used for bi- and multi-classification problems
# if dataset is non-linear separable -> rbf kernel; C: normalization parameter, adjust to optimal
# rbf: Radial Basis Function, map the data to a higher dimensional space, making the data linearly separable
clf = SVC(kernel='linear', C=1)
clf.fit(X, y)