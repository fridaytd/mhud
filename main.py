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


from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
kf = KFold(n_splits=17, shuffle=True, random_state=13)

f1_KNN = []
f1_Bayes = []
f1_Tree = []

f1_avg = {
    'KNN' : [],
    'Tree' : [],
    'Bayes' : []
}


for train_index, test_index in kf.split(X):
    #print("Train:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index,], X.iloc[test_index,]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    #print ("X_test len:", len(X_test), " X_train len:", len(X_train))
    #  KNN 
    Mohinh_KNN = KNeighborsClassifier
    Mohinh_KNN = KNeighborsClassifier(n_neighbors=7)
    Mohinh_KNN.fit(X_train, y_train)
    y_KNN_pred = Mohinh_KNN.predict(X_test)
    # Tree
    clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 50)
    clf_entropy.fit(X_train, y_train)
    y_Tree_pred = clf_entropy.predict(X_test)
    # Bayes
    Mohinh_Bayes = GaussianNB()
    Mohinh_Bayes.fit(X_train, y_train)
    y_Bayes_pred = Mohinh_Bayes.predict(X_test)
    
    #print(f1_score(y_test, y_pred, labels = ["unacc", "acc", "good", "vgood"], average=None))
    
    #  KNN
    f1 =  f1_score(y_test, y_KNN_pred, labels = ["unacc", "acc", "good", "vgood"], average=None)
    f1_KNN.append(f1)
    f1_avg['KNN'].append(sum(f1) / len(f1) * 100)
    # Tree
    f1 = f1_score(y_test, y_Tree_pred, labels = ["unacc", "acc", "good", "vgood"], average=None)
    f1_Tree.append(f1)
    f1_avg['Tree'].append(sum(f1) / len(f1) * 100)
    # Bayes
    f1 = f1_score(y_test, y_Bayes_pred, labels = ["unacc", "acc", "good", "vgood"], average=None)
    f1_Bayes.append(f1)
    f1_avg['Bayes'].append(sum(f1) / len(f1) * 100)



print("F1 KNN: ", sum(f1_KNN) / 17)
print("F1 Tree: ", sum(f1_Tree) / 17)
print("F1 Bayes: ", sum(f1_Bayes) / 17)

plt.axis([0, 18, 0, 110])

plt.plot(range(1, 18), f1_avg["KNN"], color = 'red')
plt.plot(range(1, 18), f1_avg['Tree'], color = 'green')
plt.plot(range(1, 18), f1_avg["Bayes"], color = 'blue')

# antonate

for i in range(17):
    plt.text(i + 1, f1_avg["KNN"][i], str(round(f1_avg['KNN'][i], 3)))
    plt.text(i + 1, f1_avg["Bayes"][i], str(round(f1_avg['Bayes'][i], 3)))
    plt.text(i + 1, f1_avg["Tree"][i], str(round(f1_avg['Tree'][i], 3)))
    if (i + 1 == 17):
        plt.text(i + 2, f1_avg["KNN"][i], 'KNN', color = 'red')
        plt.text(i + 2, f1_avg["Bayes"][i], 'Bayes', color = 'blue')
        plt.text(i + 2, f1_avg["Tree"][i], 'Decision Tree', color = 'green')

plt.show()