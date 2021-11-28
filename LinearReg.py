import matplotlib as plt
%matplotlib inline
import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv(r'/content/drive/MyDrive/DS PROJECTS/Linear Regression Problems/Beer consumption/Aula 0 - Iniciando o Curso.zip (Unzipped Files)/data-science/reg-linear/Dados/Consumo_cerveja.csv',
                 sep=';',
                 date_parser=True)

df.dtypes

df.describe()

linhas = df.shape[0]
variaveis = df.shape[1]
print(f'Numero de linhas={linhas}. \nNumero de variaveis={variaveis}.')

Numero de linhas=365. 
Numero de variaveis=7.

corr = df.corr()
sns.heatmap(data=corr,
            annot=True,
            fmt='.2g',
            linewidth=0.5,
            linecolor='white')

upper_tri = corr.where(np.triu(np.ones(corr.shape),k=1).astype(np.bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.85)]

X = df.drop(to_drop, axis=1)
X = X.iloc[:,1:4]
Y = df['consumo']

from matplotlib.pyplot import figure

figure(figsize=(15, 7))
ax = df['consumo'].plot()
ax.set_title('Consumo de Cerveja (L) em SP - 2015',
             fontsize=15)
ax.set_ylabel('Consumo de Cerveja (L)',
              fontsize=10)
ax.set_xlabel('Dia do ano',
              fontsize=10)

ax = sns.boxplot(data=Y,
            orient='h',
            width=0.2)
ax.figure.set_size_inches(8,5)
ax.set_title('Box plot consumo de cerveja')

ax = sns.boxplot(y=df['consumo'],
                 x=df['fds'],
            width=0.2,
            orient='v')
ax.figure.set_size_inches(5,5)

sns.pairplot(df)

pair = [x for x in X.columns]
ax = sns.pairplot(df, 
                  y_vars='consumo', 
                  x_vars=pair)
ax.figure.set_size_inches(20,7)

sns.jointplot(x='temp_max', 
              y='consumo', 
              data=df,
              kind='reg')

ax = sns.lmplot(x='temp_max', 
           y='consumo',
           data=df,
           hue='fds',
           markers=['o', '*'],
           legend=False)
ax.add_legend(title='Fim de semana')


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

lr = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    Y, 
                                                    test_size=0.3,
                                                    random_state=42)
lr.fit(X_train, y_train)

(round(lr.score(X_train, y_train),4))*100

y_predict = lr.predict(X_test)
metrics.r2_score(y_test, y_predict)
# 0.6730634560123662

lr.intercept_
6313.510265449186

lr.coef_
array([ 845.68109325,  -72.1581513 , 5382.81521322])

index=['Intercepto', 'Chuva (mm)', 'Temp. med.', 'Final de semana']
pd.DataFrame(data=np.append(lr.coef_, lr.intercept_), 
             index=index, 
             columns=['Parâmetros'])

INTERCEPTO = Efeito médio no consumo de cerveja sendo variáveis explicativas = 0.

CHUVA (mm) = Mantendo-se demais variáveis explicativas constantes, o acréscimo de 1mm gera uma variação no consumo de -72 litros.

TEMP. MED = Mantendo-se demais variáveis explicativas constantes, o acréscimo de 1 grau °C gera uma variação no consumo de 5.382,82 litros.

FINAL DE SEMANA = Mantendo-se demais variáveis explicativas constantes, o fato de ser final de semana aumenta o consumo em 6.313,51 litros.

             Parâmetros
Intercepto	845.681093
Chuva (mm)	-72.158151
Temp. med.	5382.815213
Final de semana	6313.510265

entrada = X_test[10:11]
entrada
temp_media	chuva	fds
338	24.8	0.1	1

lr.predict(entrada)[0]

temp_max = 30.5
chuva = 0.1
fds = 1
entrada=[[temp_max, chuva, fds]]
lr.predict(entrada)[0].round(2)
37482.38

MÉTRICAS DE MINIMIZAÇÃO
mse = metrics.mean_squared_error(y_test, y_predict).round(2)
msloge = metrics.mean_squared_log_error(y_test, y_predict).round(2)
msesqrt = np.sqrt(metrics.mean_squared_error(y_test, y_predict)).round(2)
r2 = metrics.r2_score(y_test, y_predict).round(2)
pd.DataFrame([mse, msloge, msesqrt, r2], 
             ['mse', 'msloge', 'msesqrt', 'r²'],
             columns=['Metricas'])

	     Metricas
mse	5560349.90
msloge	0.01
msesqrt	2358.04
r²	0.69

import pickle
output = open(r'/content/drive/MyDrive/DS PROJECTS/Linear Regression Problems/Beer consumption/Aula 0 - Iniciando o Curso.zip (Unzipped Files)/data-science/reg-linear/Projeto/modelo_consumo_cerveja',
              'wb') #write binary
pickle.dump(lr,
            output)
output.close()
