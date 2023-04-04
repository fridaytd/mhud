import pandas as pd
dt = pd.read_csv("car.data")

# amount of rows
print(len(dt))

#attributes 
attributes = dt.columns.values.tolist()
attributes.remove('evaluation')
print(attributes)

#Rows values
for attr in attributes:
    print("-----------------------------")
    print("Attribute name: ", attr)
    print(dt[attr].value_counts())

print("-----------------------------")
print("Label name: evaluation")
print(dt.evaluation.value_counts())

X = dt.iloc[:,0:6]
y = dt.evaluation
#print(X)
#print(y)

convert = {"buying": {"low":4, "med":3, "high":2, "vhigh":1},
           "maint":  {"low":4, "med":3, "high":2, "vhigh":1},
           "doors": {"2":2, "3":3, "4":4, "5more":5},
           "persons":   {"2":2, "4":4, "more":6},
           "lug_boot":  {"small":1, "med":2, "big":3},
           "safety":    {"low":1, "med":2, "high":3}}

X = X.replace(convert)

print(X)

import sklearn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.0/3, random_state=23)

from sklearn.neighbors import KNeighborsClassifier
Mohinh_KNN = KNeighborsClassifier
Mohinh_KNN = KNeighborsClassifier(n_neighbors=9)
Mohinh_KNN.fit(X_train, y_train)
# d-i.
y_pred = Mohinh_KNN.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))