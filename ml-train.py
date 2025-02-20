import warnings
warnings.simplefilter("ignore")
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import time

#%matplotlib inline

# Read the data from csv file
col_names = []
for i in range(20):
    if i == 0:
        col_names.append('quality')
    if i == 1:
        col_names.append('prescreen')
    if i >= 2 and i <= 7:
        col_names.append('ma' + str(i))
    if i >= 8 and i <= 15:
        col_names.append('exudate' + str(i))
    if i == 16:
        col_names.append('euDist')
    if i == 17:
        col_names.append('diameter')
    if i == 18:
        col_names.append('amfm_class')
    if i == 19:
        col_names.append('label')

data = pd.read_csv("messidor_features.txt", names = col_names)
print(data.shape)
data.head(10)

# 
labels = data['label']
labels.values.ravel()
data.drop(labels='label',axis=1,inplace=True)
data.head()

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
splt = train_test_split(data, test_size = 0.2, train_size = 0.8)
tree = DecisionTreeClassifier(criterion='entropy')
tree.fit(splt[0],labels[0:920])
result = tree.predict(splt[1],True)
sk.metrics.accuracy_score(labels[920:1151],result)

tree = DecisionTreeClassifier(criterion='entropy',max_depth=2)
tree.fit(splt[0],labels[0:920])
result = tree.predict(splt[1],True)
print(sk.metrics.accuracy_score(labels[920:1151],result))


tree = DecisionTreeClassifier(criterion='gini')
tree.fit(splt[0],labels[0:920])
result = tree.predict(splt[1],True)
print(sk.metrics.accuracy_score(labels[920:1151],result))



tree = DecisionTreeClassifier(criterion='entropy',min_samples_split=7)
tree.fit(splt[0],labels[0:920])
result = tree.predict(splt[1],True)
print(sk.metrics.accuracy_score(labels[920:1151],result))



tree = DecisionTreeClassifier(criterion='entropy',min_impurity_decrease=0.9)
tree.fit(splt[0],labels[0:920])
result = tree.predict(splt[1],True)
print(sk.metrics.accuracy_score(labels[920:1151],result))

tree = DecisionTreeClassifier(criterion='entropy')
accuracy = sk.model_selection.cross_val_score(tree,data,labels,cv=10)
avg = sum(accuracy)/len(accuracy)
print(str(avg))

griddy = {'max_depth' : [5,10,15,20], 'min_samples_leaf' : [5,10,15,20], 'max_features' : [5,10,15]}
gSearch = sk.model_selection.GridSearchCV(tree,griddy,scoring='accuracy')
gSearch.fit(data,labels)
print(gSearch.best_params_)
print(gSearch.best_score_)

accuracy = sk.model_selection.cross_val_score(gSearch,data,labels,cv=10)
avg = sum(accuracy)/len(accuracy)
print(str(avg))

from sklearn.naive_bayes import GaussianNB
gauss = sk.naive_bayes.GaussianNB()
gauss.fit(data,labels)

accuracy = sk.model_selection.cross_val_score(gauss,data,labels,cv=10)
avg = sum(accuracy)/len(accuracy)
print(str(avg))

pred = sk.model_selection.cross_val_predict(gauss,data,labels)
matrix = sk.metrics.confusion_matrix(labels,pred)
print(matrix)
print(sk.metrics.classification_report(labels,pred))

splt = train_test_split(data, test_size = 0.2, train_size = 0.8)

gauss = sk.naive_bayes.GaussianNB()
gauss.fit(splt[0],labels[0:920])

prob = gauss.predict_proba(splt[1])

roc = sk.metrics.roc_curve(labels[920:1151],prob[:,1])

fpr, tpr = roc[0], roc[1]

auc = sk.metrics.roc_auc_score(labels[920:1151],prob[:,1])
print(auc)

plt.plot([0,1],[0,1],'k--') 
plt.plot(fpr, tpr, label='NB') 
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC Curve Naive Bayes')
plt.show()

from sklearn.pipeline import Pipeline
scaler = sk.preprocessing.StandardScaler()
PCA = sk.decomposition.PCA()
KN = sk.neighbors.KNeighborsClassifier(n_neighbors=7)
pipe = sk.pipeline.Pipeline([('scale',scaler),('pca',PCA),('kn',KN)])
accuracy = sk.model_selection.cross_val_score(pipe,data,labels)
avg = sum(accuracy)/len(accuracy)
print(str(avg))

param_grid = {
    'pca__n_components': list(range(5, 19)),
    'kn__n_neighbors': list(range(1, 25))
}

gSearch2 = sk.model_selection.GridSearchCV(pipe,param_grid,cv=5)
gSearch2.fit(data,labels)
print(gSearch2.best_params_)
print(gSearch2.best_score_)

accuracy = sk.model_selection.cross_val_score(gSearch2,data,labels)
avg = sum(accuracy)/len(accuracy)
print(str(avg))

scaler = sk.preprocessing.StandardScaler()
PCA = sk.decomposition.PCA()
SVC = sk.svm.SVC()
pipe = sk.pipeline.Pipeline([('scale',scaler),('pca',PCA),('svc',SVC)])

param_grid = {
    'pca__n_components': list(range(5, 19)),
    'svc__kernel': ['linear','rbf','poly']
}

gSearch3 = sk.model_selection.GridSearchCV(pipe,param_grid,cv=5)
gSearch3.fit(data,labels)

accuracy = sk.model_selection.cross_val_score(gSearch3,data,labels)
avg = sum(accuracy)/len(accuracy)
print(str(avg))

from sklearn.neural_network import MLPClassifier
neural = sk.neural_network.MLPClassifier()
pipe = sk.pipeline.Pipeline([('scale',scaler),('neural',neural)])

param_grid = {
    'neural__hidden_layer_sizes': [30,40,50,60],
    'neural__activation': ['logistic','tanh','relu']
}

gSearch4 = sk.model_selection.GridSearchCV(pipe,param_grid,cv=5)

accuracy = sk.model_selection.cross_val_score(gSearch4,data,labels)
avg = sum(accuracy)/len(accuracy)
print(str(avg))

from sklearn.ensemble import RandomForestClassifier
forest = sk.ensemble.RandomForestClassifier()

param_grid = {
    'n_estimators': [50,100,150]
}

forestSearch = sk.model_selection.GridSearchCV(forest,param_grid,cv=5)
accuracy = sk.model_selection.cross_val_predict(forestSearch,data,labels)

print(sk.metrics.classification_report(labels,pred))

from sklearn.ensemble import AdaBoostClassifier
forest = sk.ensemble.AdaBoostClassifier()

param_grid = {
    'n_estimators': [50,100,150]
}

forestSearch = sk.model_selection.GridSearchCV(forest,param_grid,cv=5)
pred = sk.model_selection.cross_val_predict(forestSearch,data,labels)
print(sk.metrics.classification_report(labels,pred))

import pickle

scaler = sk.preprocessing.StandardScaler()
PCA = sk.decomposition.PCA()
SVC = sk.svm.SVC()
pipe = sk.pipeline.Pipeline([('scale',scaler),('pca',PCA),('svc',SVC)])

param_grid = {
    'pca__n_components': list(range(5, 19)),
    'svc__kernel': ['linear','rbf','poly']
}

gSearch3 = sk.model_selection.GridSearchCV(pipe,param_grid,cv=5)
gSearch3.fit(data,labels)

print(gSearch3.best_params_)

PCA_F = sk.decomposition.PCA(n_components = 18)
SVC_F = sk.svm.SVC(kernel = 'linear')
pipe_F = sk.pipeline.Pipeline([('scale',scaler),('pca',PCA_F),('svc',SVC_F)])
pipe_F.fit(data,labels)
final_model = pipe_F

filename = 'finalized_model.sav'
pickle.dump(final_model, open(filename, 'wb'))

record = [ 0.05905386, 0.2982129, 0.68613149, 0.75078865, 0.87119216, 0.88615694,
  0.93600623, 0.98369184, -0.47426472, -0.57642756, -0.53115361, -0.42789774,
 -0.21907738, -0.20090532, -0.21496782, -0.2080998, 0.06692373, -2.81681183,
 -0.7117194 ]

 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

result = loaded_model.predict([record])
if(result == 1):
    print('Positive for disease')
else:
    print('Negative for disease')