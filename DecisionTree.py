import pandas as pd

#Get data
dt = pd.read_csv("car.data")

X = dt.iloc[:,0:6]
y = dt.evaluation
#print(X)
#print(y)

#Convert attributes
convert = {"buying": {"low":4, "med":3, "high":2, "vhigh":1},
           "maint":  {"low":4, "med":3, "high":2, "vhigh":1},
           "doors": {"2":2, "3":3, "4":4, "5more":5},
           "persons":   {"2":2, "4":4, "more":6},
           "lug_boot":  {"small":1, "med":2, "big":3},
           "safety":    {"low":1, "med":2, "high":3}}

X = X.replace(convert)
#print(X)

from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
kf = KFold(n_splits=17, shuffle=True)

#X = data.iloc[:,1:4]
#y = data.iloc[:,4:5]
avg = [0.0, 0.0, 0.0, 0.0];
for train_index, test_index in kf.split(X):
    #print("Train:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index,], X.iloc[test_index,]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    #print ("X_test len:", len(X_test), " X_train len:", len(X_train))
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 50)
    clf_entropy.fit(X_train, y_train)
    
    y_pred = clf_entropy.predict(X_test)
    #print(f1_score(y_test, y_pred, labels = ["unacc", "acc", "good", "vgood"], average=None))
    
    f1 = f1_score(y_test, y_pred, labels = ["unacc", "acc", "good", "vgood"], average=None)
    avg[0] = avg[0] + f1[0]
    avg[1] = avg[1] + f1[1]
    avg[2] = avg[2] + f1[2]
    avg[3] = avg[3] + f1[3]
    #print("=======================")
    
print(avg[0]/17)
print(avg[1]/17)
print(avg[2]/17)
print(avg[3]/17)