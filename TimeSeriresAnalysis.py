import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

alucar = pd.read_csv('/content/alucar.csv', sep=',')
alucar.rename(columns={'mes': 'data'}, inplace=True)
alucar['ano'] = pd.DatetimeIndex(alucar['data'], dayfirst=True).year
alucar['mes'] = pd.DatetimeIndex(alucar['data'], dayfirst=True).month
alucar['dia'] = pd.DatetimeIndex(alucar['data'], dayfirst=True).day

dw_mapping={
    0: 'Segunda-feira', 
    1: 'Terça-feira', 
    2: 'Quarta-feira', 
    3: 'Quinta-feira', 
    4: 'Sexta-feira',
    5: 'Sábado', 
    6: 'Domingo'
} 

alucar['dia_da_semana']= pd.DatetimeIndex(alucar['data']).weekday.map(dw_mapping)
alucar['data'] = pd.to_datetime(alucar['data'])
alucar['crescimento'] = round(alucar['vendas'].pct_change(), 4) * 100
alucar['variacao_vendas'] = alucar['vendas'].diff()
alucar['aceleracao'] = alucar['variacao_vendas'].diff()
alucar = alucar[['data', 'ano', 'mes', 'dia', 'dia_da_semana', 'vendas', 'crescimento', 'variacao_vendas', 'aceleracao']]

sns.set_palette('Accent')
sns.set_style('darkgrid')
ax = sns.lineplot(x='data', y='vendas', data=alucar)
ax.figure.set_size_inches(12,6)
ax.set_title('Vendas Alucar de 2017 e 2018', loc='left', fontsize=18)
ax.set_xlabel('Tempo', fontsize=14)
ax.set_ylabel('Vendas (R$)', fontsize=14)
ax = ax

sns.set_palette('Accent')
sns.set_style('darkgrid')
ax2 = sns.lineplot(x='data', y='crescimento', data=alucar)
ax2.figure.set_size_inches(12,6)
ax2.set_title('Crescimento da Alucar ao longo dos anos', loc='left', fontsize=18)
ax2.set_xlabel('Tempo', fontsize=14)
ax2.set_ylabel('Crescimento %', fontsize=14)
ax2 = ax2

sns.set_palette('Accent')
sns.set_style('darkgrid')
ax3 = sns.lineplot(x='data', y='variacao_vendas', data=alucar)
ax3.figure.set_size_inches(12,6)
ax3.set_title('Variacao das Vendas da Alucar ao longo dos anos', loc='left', fontsize=18)
ax3.set_xlabel('Tempo', fontsize=14)
ax3.set_ylabel('Variacao Vendas (R$)', fontsize=14)
ax3= ax3

sns.set_palette('Accent')
sns.set_style('darkgrid')
ax4 = sns.lineplot(x='data', y='aceleracao', data=alucar)
ax4.figure.set_size_inches(12,6)
ax4.set_title('Crescimento da Alucar ao longo dos anos', loc='left', fontsize=18)
ax4.set_xlabel('Tempo', fontsize=14)
ax4.set_ylabel('Aceleracao', fontsize=14)
ax4= ax4

plt.figure(figsize=(16,12))
ax = plt.subplot(3,1,1)
ax.set_title('Analise do crescimento da Alucar', fontsize=18, loc='left')
sns.lineplot(x='data', y='vendas', data=alucar)
ax = plt.subplot(3,1,2)
sns.lineplot(x='data', y='variacao_vendas', data=alucar)
ax = plt.subplot(3,1,3)
sns.lineplot(x='data', y='aceleracao', data=alucar)
ax = ax

from pandas.plotting import autocorrelation_plot

autocorrelation_plot(alucar['vendas'])

autocorrelation_plot(alucar['variacao_vendas'][1:])

autocorrelation_plot(alucar['aceleracao'][3:]) #índice que começa os dados (3.0)

newsletter = pd.read_csv('/content/newsletter_alucar.csv', sep=',')
newsletter.rename(columns={'mes': 'data'}, inplace=True)
newsletter['ano'] = pd.DatetimeIndex(newsletter['data'], dayfirst=True).year
newsletter['mes'] = pd.DatetimeIndex(newsletter['data'], dayfirst=True).month
newsletter['dia'] = pd.DatetimeIndex(newsletter['data'], dayfirst=True).day
newsletter['num_da_semana'] = pd.DatetimeIndex(newsletter['data'], dayfirst=True).week
newsletter['data'] = pd.to_datetime(newsletter['data'])
newsletter['crescimento'] = round(newsletter['assinantes'].pct_change(), 4) * 100
newsletter['variacao_assinantes'] = newsletter['assinantes'].diff()
newsletter['aceleracao'] = newsletter['assinantes'].diff()
newsletter = newsletter[['data', 'ano', 'mes', 'dia', 'num_da_semana', 'assinantes', 
                         'crescimento', 'variacao_assinantes', 'aceleracao']]

chocolura = pd.read_csv('/content/chocolura.csv')
chocolura.rename(columns={'mes': 'data'}, inplace=True)
chocolura['ano'] = pd.DatetimeIndex(chocolura['data'], dayfirst=True).year
chocolura['mes'] = pd.DatetimeIndex(chocolura['data'], dayfirst=True).month
chocolura['dia'] = pd.DatetimeIndex(chocolura['data'], dayfirst=True).day
chocolura['num_da_semana'] = pd.DatetimeIndex(chocolura['data'], dayfirst=True).week
chocolura['data'] = pd.to_datetime(chocolura['data'])
chocolura['crescimento'] = round(chocolura['vendas'].pct_change(), 4) * 100
chocolura['variacao_vendas'] = chocolura['vendas'].diff()
chocolura['aceleracao'] = chocolura['variacao_vendas'].diff()
chocolura = chocolura[['data', 'ano', 'mes', 'dia', 'num_da_semana', 
                       'vendas', 'crescimento', 'variacao_vendas', 'aceleracao']]

chocolura_dia = pd.read_csv('/content/vendas_por_dia.csv')
chocolura_dia.rename(columns={'dia': 'data'}, inplace=True)
chocolura_dia['ano'] = pd.DatetimeIndex(chocolura_dia['data'], dayfirst=True).year
chocolura_dia['mes'] = pd.DatetimeIndex(chocolura_dia['data'], dayfirst=True).month
chocolura_dia['dia'] = pd.DatetimeIndex(chocolura_dia['data'], dayfirst=True).day
chocolura_dia['num_da_semana'] = pd.DatetimeIndex(chocolura_dia['data'], dayfirst=True).week
chocolura_dia['data'] = pd.to_datetime(chocolura_dia['data'])
chocolura_dia['crescimento'] = round(chocolura_dia['vendas'].pct_change(), 4) * 100
chocolura_dia['variacao_vendas'] = chocolura_dia['vendas'].diff()
chocolura_dia['aceleracao'] = chocolura_dia['variacao_vendas'].diff()
dia_da_semana_mapping={
    0: 'Segunda-feira', 
    1: 'Terça-feira', 
    2: 'Quarta-feira', 
    3: 'Quinta-feira', 
    4: 'Sexta-feira',
    5: 'Sábado', 
    6: 'Domingo'
} 
chocolura_dia['dia_da_semana']= pd.DatetimeIndex(chocolura_dia['data']).weekday.map(dw_mapping)
chocolura_dia = chocolura_dia[['data', 'ano', 'mes', 'dia', 'dia_da_semana', 'num_da_semana', 
                       'vendas', 'crescimento', 'variacao_vendas', 'aceleracao']]

plt.figure(figsize=(12,6))
sns.boxplot(x='dia_da_semana', y='vendas', data=chocolura_dia)

vendas_agrupadas = chocolura_dia.groupby('dia_da_semana')[['vendas','crescimento', 'variacao_vendas', 'aceleracao']].mean().round()
autocorrelation_plot(chocolura_dia['vendas'][1:])

cafelura = pd.read_csv('/content/cafelura.csv')
cafelura.rename(columns={'mes': 'data'}, inplace=True)
cafelura['ano'] = pd.DatetimeIndex(cafelura['data'], dayfirst=True).year
cafelura['mes'] = pd.DatetimeIndex(cafelura['data'], dayfirst=True).month
cafelura['dia'] = pd.DatetimeIndex(cafelura['data'], dayfirst=True).day
cafelura['num_da_semana'] = pd.DatetimeIndex(cafelura['data'], dayfirst=True).week
cafelura['data'] = pd.to_datetime(cafelura['data'])
cafelura['crescimento'] = round(cafelura['vendas'].pct_change(), 4) * 100
cafelura['variacao_vendas'] = cafelura['vendas'].diff()
cafelura['aceleracao'] = cafelura['variacao_vendas'].diff()
cafelura = cafelura[['data', 'ano', 'mes', 'dia', 'num_da_semana', 
                       'vendas', 'crescimento', 'variacao_vendas', 'aceleracao']]

dias_final_de_semana = pd.read_csv('/content/dias_final_de_semana.csv')
dias_final_de_semana['quantidade_de_dias'].values

from statsmodels.tsa.seasonal import seasonal_decompose

resultado = seasonal_decompose([chocolura['vendas']], freq=2)
ax = resultado.plot()

observed = resultado.observed
trend = resultado.trend
seasonal = resultado.seasonal
residual = resultado.resid
data = ({
    'observed': observed,
    'trend': trend,
    'seasonzal': seasonal,
    'residual': residual,
})
resultado = pd.DataFrame(data)
resultado.head()

alucel = pd.read_csv('/content/alucel.csv')
alucel['dia'] = pd.to_datetime(alucel['dia'])
alucel.isna().sum()

alucel['aumento'] = alucel['vendas'].diff()
alucel['aceleracao'] = alucel['aumento'].diff()

def plot_comparacao(x, y1, y2, y3, dataset, titulo):
  plt.figure(figsize=(16,12))
  ax = plt.subplot(3,1,1)
  ax.set_title(titulo, fontsize=18,loc='left')
  sns.lineplot(x=x, y=y1, data=dataset)
  plt.subplot(3,1,2)
  sns.lineplot(x=x, y=y2, data=dataset)
  plt.subplot(3,1,3)
  sns.lineplot(x=x, y=y3, data=dataset)
  ax = ax
  
  plot_comparacao(x='dia', y1='vendas', y2='aumento', y3='aceleracao', 
                dataset=alucel, 
                titulo='Analise das Vendas da Alucel de Outubro a Novembro de 2018')
 
alucel['media_movel'] = alucel['vendas'].rolling(7).mean()
alucel.head(14)

plt.figure(figsize=(12,6))
ax = sns.lineplot(data=alucel, x='dia', y='media_movel')
ax.set_title('Media movel das Vendas (R$) de Outubro a Novembro de 2018', fontsize=18)
ax.set_xlabel('DIA', fontsize=14)
ax.set_ylabel('MEDIA MOVEL', fontsize=14)
ax = ax
