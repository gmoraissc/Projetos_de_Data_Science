import pandas as pd
import numpy as np

from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

uri = "https://gist.githubusercontent.com/guilhermesilveira/e99a526b2e7ccc6c3b70f53db43a87d2/raw/1605fc74aa778066bf2e6695e24d53cf65f2f447/machine-learning-carros-simulacao.csv"

dados = pd.read_csv(uri).drop(columns=["Unnamed: 0"], axis=1)

x = dados.loc[:, dados.columns != 'vendido']
y = dados["vendido"]

x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.25,
                                                    stratify=y)

from sklearn.model_selection import GridSearchCV

espaco_de_parametros = {
    'max_depth': [3, 5],
    'min_samples_split': [32, 64, 128],
    'min_samples_leaf': [32, 64, 128],
    'criterion': ['gini', 'entropy']
}

busca = GridSearchCV(DecisionTreeClassifier(),
                     espaco_de_parametros,
                     cv= GroupKFold(n_splits=2))

busca.fit(x, y, groups=dados.vendido)
resultados = pd.DataFrame(busca.cv_results_)

busca.best_params_

melhor = busca.best_estimator_

predicoes = melhor.predict(x_test)
acuracia = accuracy_score(predicoes, y_test)
acuracia
#evitar esta abordagem, é muito otimista para cross validation

NESTED CROSS VALIDATION

from sklearn.model_selection import cross_val_score, KFold

espaco_de_parametros = {
    'max_depth': [3, 5],
    'min_samples_split': [32, 64, 128],
    'min_samples_leaf': [32, 64, 128],
    'criterion': ['gini', 'entropy']
}

busca = GridSearchCV(DecisionTreeClassifier(),
                     espaco_de_parametros,
                     cv= KFold(n_splits=2))

busca.fit(x, y)
resultados = pd.DataFrame(busca.cv_results_)
cross_val_score(busca, 
                x, 
                y, 
                cv= KFold(n_splits=2, shuffle=True), 
                groups=dados.vendido)
array([0.7922, 0.7824])

ESTIMADORES DE MÍNIMOS QUADRADOS
import statsmodels.api as sm
y = dados.vendido
X = sm.add_constant(x)
resultados_reg = sm.OLS(y, x, missing='drop').fit()
resultados_reg.params
resultados_reg.conf_int(alpha=0.05) #0 = inferior, 1 = superior

	                    0	       1
preco	          -0.000003	-0.000002
idade_do_modelo	 0.033759	0.037013
km_por_ano	     0.000007	0.000009
