import pandas as pd

uri = "https://gist.githubusercontent.com/guilhermesilveira/e99a526b2e7ccc6c3b70f53db43a87d2/raw/1605fc74aa778066bf2e6695e24d53cf65f2f447/machine-learning-carros-simulacao.csv"

dados = pd.read_csv(uri).drop(columns=["Unnamed: 0"], axis=1)

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

y = dados['vendido']
x = dados.loc[:, dados.columns != 'vendido']

seed = 42
np.random.seed(seed)
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    test_size=0.25,
                                                    stratify=y)

ESTABELECENDO UMA NOTA DE CORTE PARA SCORE (DUMMY CLASSIFIER)
dummy_stratified = DummyClassifier()
dummy_stratified.fit(x_train, y_train)
acuracia = dummy_stratified.score(x_test, y_test)
acuracia
0.58

from sklearn.tree import DecisionTreeClassifier
modelo = DecisionTreeClassifier(max_depth=2)
modelo.fit(x_train, y_train)
previsoes = modelo.predict(x_test)
acuracia = accuracy_score(y_test, previsoes)
acuracia
0.7512

CROSS VALIDATE (VALIDACAO CRUZADA)
from sklearn.model_selection import cross_validate

modelo = DecisionTreeClassifier(max_depth=2)
resultados = cross_validate(modelo, 
                            x, 
                            y, 
                            cv=10,
                            return_train_score=False)
media = resultados['test_score'].mean()
desv_pad = resultados['test_score'].std()
intervalo = media -1.96 * desv_pad, media + 1.96 * desv_pad
intervalo
(0.7427245545339448, 0.772875445466055)

from sklearn.model_selection import KFold #splitter classes

def imprime_resultados(resultados):
    media = resultados['test_score'].mean()

    desv_pad = resultados['test_score'].std()

    intervalo = media -1.96 * desv_pad, media + 1.96 * desv_pad
    print(intervalo)

cv = KFold(n_splits=10, shuffle=True)

modelo = DecisionTreeClassifier(max_depth=2)

resultados = cross_validate(modelo, 
                            x, 
                            y, 
                            cv=cv,
                            return_train_score=False)

imprime_resultados(resultados)
(0.7303628001428716, 0.7852371998571283)

from sklearn.model_selection import StratifiedKFold #splitter classes

def imprime_resultados(resultados):
    media = resultados['test_score'].mean()

    desv_pad = resultados['test_score'].std()

    intervalo = media -1.96 * desv_pad, media + 1.96 * desv_pad
    print(intervalo)

cv = StratifiedKFold(n_splits=10, shuffle=True)

modelo = DecisionTreeClassifier(max_depth=2)

resultados = cross_validate(modelo, 
                            x, 
                            y, 
                            cv=cv,
                            return_train_score=False)

imprime_resultados(resultados)
(0.7400774376570429, 0.7755225623429571)

VALIDACAO CRUZADA COM GRUPOS
from sklearn.model_selection import GroupKFold

cv = GroupKFold(n_splits=10)

modelo = DecisionTreeClassifier(max_depth=7)

resultados = cross_validate(modelo, 
                            x, 
                            y, 
                            cv=cv,
                            return_train_score=False,
                            groups=dados['idade_do_modelo'])

imprime_resultados(resultados)
(0.7313421865188141, 0.8294646492795452)

PIPELINE - STANDARD SCALER
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

from sklearn.svm import SVC
modelo = SVC()
modelo.fit(x_train_scaled, y_train)
previsoes = modelo.predict(x_test_scaled)
acuracia = accuracy_score(y_test, previsoes)
acuracia
0.7604

from sklearn.model_selection import GroupKFold

cv = GroupKFold(n_splits=10)

modelo = SVC()

resultados = cross_validate(modelo, 
                            x, 
                            y, 
                            cv=cv,
                            return_train_score=False,
                            groups=dados['idade_do_modelo'])

imprime_resultados(resultados)
(0.7185685284595239, 0.8258023083011053)

from sklearn.pipeline import Pipeline

scaler = StandardScaler()
modelo = SVC()

pipeline = Pipeline(
    [
     ('transformacao',scaler),
    ('estimador',modelo)
    ]
)

cv = GroupKFold(n_splits=10)

resultados = cross_validate(pipeline, 
                            x, 
                            y, 
                            cv=cv,
                            return_train_score=False,
                            groups=dados['idade_do_modelo'])

imprime_resultados(resultados)

scaler = StandardScaler()
modelo = SVC()

pipeline = Pipeline(
    [
     ('transformacao',scaler),
    ('estimador',modelo)
    ]
)

cv = GroupKFold(n_splits=10)

resultados = cross_validate(pipeline, 
                            x, 
                            y, 
                            cv=cv,
                            return_train_score=False,
                            groups=dados['idade_do_modelo'])

imprime_resultados(resultados)
(0.7213695992808791, 0.8138048002750555)
