import pandas as pd
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv('credit_data.csv')

columns = dataset.columns
for i in dataset[columns]:
  sns.countplot(dataset[i])
  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
modelo = GaussianNB()
modelo.fit(X_train, y_train)
predictions = modelo.predict(X_test)
accuracy_score(predictions, y_test)
0.915

cm = confusion_matrix(predictions, y_test)
cm
array([[334,  25],
       [  9,  32]])

from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTE
tl = TomekLinks(return_indices=True, ratio='majority')
X_under, y_under, id_under = tl.fit_sample(X, y)

X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(X_under,
                                                            y_under,
                                                            test_size=0.2,
                                                            stratify=y_under)

modelo_u = GaussianNB()
modelo_u.fit(X_train_u, y_train_u)
predictions_u = modelo.predict(X_test_u)
accuracy_score(predictions_u, y_test_u)
0.9210526315789473

cm_u = confusion_matrix(predictions_u, y_test_u)
cm_u
array([[316,  23],
       [  7,  34]])

smote = SMOTE(ratio='minority')

X_train_o, X_test_o, y_train_o, y_test_o = train_test_split(X_over, y_over,
                                                            test_size=0.2,
                                                            stratify=y_over)

modelo_o = GaussianNB()
modelo_o.fit(X_train_o, y_train_o)
predictions_o = modelo_o.predict(X_test_o)
accuracy_score(predictions_o, y_test_o)

cm_o = confusion_matrix(predictions_o, y_test_o)
cm_o
array([[300,  10],
       [ 43, 333]])

