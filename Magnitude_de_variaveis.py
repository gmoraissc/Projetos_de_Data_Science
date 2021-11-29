import pandas as pd
import numpy as np

# import several machine learning algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

# to scale the features
from sklearn.preprocessing import MinMaxScaler

# to evaluate performance and separate into
# train and test set
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

titanic = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl',
                   usecols=['pclass', 'age', 'fare', 'survived'])
titanic = titanic.replace('?', np.nan)

titanic.dropna(axis=0, inplace=True)
titanic['age'] = pd.to_numeric(titanic['age'], downcast='integer')
titanic['fare'] = pd.to_numeric(titanic['fare'], downcast='integer')

X_train, X_test, y_train, y_test = train_test_split(
    titanic[['pclass', 'age', 'fare']].fillna(0),
    titanic.survived,
    test_size=0.3,
    random_state=0)

# call the scaler
scaler = MinMaxScaler()

# fit the scaler
scaler.fit(X_train)

# re scale the datasets
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print('Mean: ', X_train_scaled.mean(axis=0))
print('Standard Deviation: ', X_train_scaled.std(axis=0))
print('Minimum value: ', X_train_scaled.min(axis=0))
print('Maximum value: ', X_train_scaled.max(axis=0))

# call the model
logit = LogisticRegression(
    random_state=44,
    C=1000,  # c big to avoid regularization
    solver='lbfgs')

# train the model
logit.fit(X_train, y_train)

# evaluate performance
print('Train set')
pred = logit.predict_proba(X_train)
print('Logistic Regression roc-auc: {}'.format(
    roc_auc_score(y_train, pred[:, 1])))
print('Test set')
pred = logit.predict_proba(X_test)
print('Logistic Regression roc-auc: {}'.format(
    roc_auc_score(y_test, pred[:, 1])))

Train set
Logistic Regression roc-auc: 0.7172140120415983
Test set
Logistic Regression roc-auc: 0.725358610854794
  
  logit.coef_
  array([[-0.93739133, -0.03235682,  0.00404752]])
  
  # model built on scaled variables

# call the model
logit = LogisticRegression(
    random_state=44,
    C=1000,  # c big to avoid regularization
    solver='lbfgs')

# train the model using the re-scaled data
logit.fit(X_train_scaled, y_train)

# evaluate performance
print('Train set')
pred = logit.predict_proba(X_train_scaled)
print('Logistic Regression roc-auc: {}'.format(
    roc_auc_score(y_train, pred[:, 1])))
print('Test set')
pred = logit.predict_proba(X_test_scaled)
print('Logistic Regression roc-auc: {}'.format(
    roc_auc_score(y_test, pred[:, 1])))

Train set
Logistic Regression roc-auc: 0.7172140120415982
Test set
Logistic Regression roc-auc: 0.7254005536448285
  
Support Vector Machines
# model build on unscaled variables

# call the model
SVM_model = SVC(random_state=44, probability=True, gamma='auto')

#  train the model
SVM_model.fit(X_train, y_train)

# evaluate performance
print('Train set')
pred = SVM_model.predict_proba(X_train)
print('SVM roc-auc: {}'.format(roc_auc_score(y_train, pred[:, 1])))
print('Test set')
pred = SVM_model.predict_proba(X_test)
print('SVM roc-auc: {}'.format(roc_auc_score(y_test, pred[:, 1])))

Train set
SVM roc-auc: 0.8848815388224255
Test set
SVM roc-auc: 0.6240667729217347
  
  # model built on scaled variables

# call the model
SVM_model = SVC(random_state=44, probability=True, gamma='auto')

# train the model
SVM_model.fit(X_train_scaled, y_train)

# evaluate performance
print('Train set')
pred = SVM_model.predict_proba(X_train_scaled)
print('SVM roc-auc: {}'.format(roc_auc_score(y_train, pred[:, 1])))
print('Test set')
pred = SVM_model.predict_proba(X_test_scaled)
print('SVM roc-auc: {}'.format(roc_auc_score(y_test, pred[:, 1])))

Train set
SVM roc-auc: 0.7084643052623348
Test set
SVM roc-auc: 0.7144954282358863
  
K-Nearest Neighbors
#model built on unscaled features

# call the model
KNN = KNeighborsClassifier(n_neighbors=5)

# train the model
KNN.fit(X_train, y_train)

# evaluate performance
print('Train set')
pred = KNN.predict_proba(X_train)
print('KNN roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
print('Test set')
pred = KNN.predict_proba(X_test)
print('KNN roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))

Train set
KNN roc-auc: 0.8208382203456094
Test set
KNN roc-auc: 0.6375723513128093
  
# model built on scaled

# call the model
KNN = KNeighborsClassifier(n_neighbors=5)

# train the model
KNN.fit(X_train_scaled, y_train)

# evaluate performance
print('Train set')
pred = KNN.predict_proba(X_train_scaled)
print('KNN roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
print('Test set')
pred = KNN.predict_proba(X_test_scaled)
print('KNN roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))

Train set
KNN roc-auc: 0.8487371960278364
Test set
KNN roc-auc: 0.7047856723429242
  
Random Forests
# model built on unscaled features

# call the model
rf = RandomForestClassifier(n_estimators=200, random_state=39)

# train the model
rf.fit(X_train, y_train)

# evaluate performance
print('Train set')
pred = rf.predict_proba(X_train)
print('Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred[:, 1])))
print('Test set')
pred = rf.predict_proba(X_test)
print('Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred[:, 1])))

Train set
Random Forests roc-auc: 0.9973922902494331
Test set
Random Forests roc-auc: 0.6949920308698935
  
  # model built in scaled features

# call the model
rf = RandomForestClassifier(n_estimators=200, random_state=39)

# train the model
rf.fit(X_train_scaled, y_train)

# evaluate performance
print('Train set')
pred = rf.predict_proba(X_train_scaled)
print('Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
print('Test set')
pred = rf.predict_proba(X_test_scaled)
print('Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))

Train set
Random Forests roc-auc: 0.9974313863476425
Test set
Random Forests roc-auc: 0.6940692894891369
  
Adaboost
# train adaboost on non-scaled features

# call the model
ada = AdaBoostClassifier(n_estimators=200, random_state=44)

# train the model
ada.fit(X_train, y_train)

# evaluate model performance
print('Train set')
pred = ada.predict_proba(X_train)
print('AdaBoost roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
print('Test set')
pred = ada.predict_proba(X_test)
print('AdaBoost roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))

Train set
AdaBoost roc-auc: 0.8265501602940026
Test set
AdaBoost roc-auc: 0.7183961077090848
  
# train adaboost on scaled features

# call the model
ada = AdaBoostClassifier(n_estimators=200, random_state=44)

# train the model
ada.fit(X_train_scaled, y_train)

# evaluate model performance
print('Train set')
pred = ada.predict_proba(X_train_scaled)
print('AdaBoost roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
print('Test set')
pred = ada.predict_proba(X_test_scaled)
print('AdaBoost roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))

Train set
AdaBoost roc-auc: 0.8265501602940026
Test set
AdaBoost roc-auc: 0.7183961077090848
