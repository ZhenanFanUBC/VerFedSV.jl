from libsvm.svmutil import *
import sklearn
import numpy as np
import shap
from scipy import sparse

# load data
filepath = './data/Adult/a8a'
Y, X = svm_read_problem(filepath)
I = []
J = []
V = []
N = len(X)
for i in range(N):
    for j in X[i].keys():
        I.append(i)
        J.append(j-1)
        V.append(X[i][j])
X = sparse.coo_matrix((V,(I,J)),shape=(N,123)).tocsr()
Y = np.array(Y)

# model
model = sklearn.linear_model.LogisticRegression(max_iter=1000, penalty='l2')
print("start traing logistic regression")
model.fit(X, Y)
print("finish traing")

# accuracy
N = len(Y)
acc = 0.0
for i in range(N):
    pred = model.predict(X[i,:])[0]
    if pred == Y[i]:
        acc += 1/N
print("training accuracy = ", acc)

# SHAP
print("create explainer")
X = X.toarray()
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# feature importance
print("compute feature importance")
d = len(shap_values.values[0])
feature_importance = np.zeros(d)
for i in range(N):
    for j in range(d):
        feature_importance[j] += shap_values.values[i][j]/N
for j in range(d):
    print("feature ", j, "value", feature_importance[j])

# group importance
print("compute group importance")
k = 3
dm = int(d/k)
group_importance = np.zeros(k)
for i in range(k):
    group_importance[i] = sum(feature_importance[i*dm: (i+1)*dm])
print(group_importance)
