import pickle
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier



iris = datasets.load_iris()
X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
model.fit(X_train, y_train)

model.fit(X_train, y_train)
# save model to file
pickle.dump(model, open("model.pkl", "wb"))

print(X_train)
print(y_train)


