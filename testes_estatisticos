import pandas as pd
import seaborn as sns
import numpy as np
from statsmodels.stats.weightstats import DescrStatsW
from statsmodels.stats.weightstats import zconfint
from statsmodels.stats.weightstats import ztest
from scipy.stats import ttest_ind
from scipy.stats import normaltest
from scipy.stats import wilcoxon, ranksums
from scipy.stats import norm
from scipy.stats import t


df = pd.read_csv(r'/content/drive/MyDrive/DS PROJECTS/Linear Regression Problems/House prices/HousePrices_HalfMil.csv')

CUMULATIVE DISTRIBUTION FUNCTION (CDF)

Dado o tipo de amostra (populacional ou amostral), a partir do z-score/t-score, qual é a área de probabilidade de dada posição na distribuição?

sns.distplot(df.Prices, 
             hist_kws={'cumulative': True},
                                  kde_kws={'cumulative':True})
                                  
df.query('Prices>0').Prices.dropna().quantile(0.8)

#52725.0

df.query('Prices>0').Prices.dropna().quantile(0.2)

#31450.0

INTERVALO DE CONFIANÇA

int_i = zconfint(df.Prices)[0]
int_f = zconfint(df.Prices)[1]
print(f"O preço das casas está entre: {int_i} e {int_f}")

desc = DescrStatsW(df.Prices)
desc.tconfint_mean(alpha=0.05,alternative='two-sided')

#Se ztest p value menor ou igual 0.05, descartar hipótese.

ztest(df.Prices, value=42500) # Neste caso, devemos descartar a hipótese de que a média é igual a 42500.

Comparando médias

df1 = df.query('Area >= 150')
zconfint(df1.Prices, df.Prices)
#(1793.5502249368076, 1918.7053274502612)

#As casas com pelo menos 150 m² possuem um preço entre 1793 e 1918 mais alto do que as demais casas.
ztest(df1.Prices, df.Prices) #descartar hipótese de que as casas com mais do que 150m² possuem preço médio maior

zconfint(df.Prices, df1.Prices) #os preços médios gerais são menores que os preços das casas com pelo menos 150m²

ttest_ind(df.Prices, df1.Prices)
#Ttest_indResult(statistic=-58.13496244234528, pvalue=0.0)

df1 = df.query('Area >= 150')
df2 = df.query('Area < 150')
house1 = DescrStatsW(df1.Prices)
house2 = DescrStatsW(df2.Prices)
comparacao = house1.get_compare(house2)
comparacao.summary()

Test for equality of means
               coef	std err	t	      P>|t|	  [0.025	0.975]
subset #1	3097.1700	34.674	89.324	0.000	3029.211	3165.129

sns.distplot(df.Prices)

#testa a hipótese nula de que a amostra vem de uma distribuição normal, logo
# descartar hipótese nula se p <=0.05, logo não vem de uma distribuição normal, deve-se aplicar testes não-paramétricos

stat, p = normaltest(df.Prices)

_, p = ranksums(df1.Prices, df2.Prices)
p #duas amostras vêm da mesma distribuição = h0, se p <=0.05, rejeitá-la

_, p = wilcoxon(df1.Prices[0:31], df2.Prices[0:31])
p
#0.09577192087902134

TESTE DE HIPÓTESES

H1 = Alvo de estudo, hipótese alternativa, deve definir uma desigualdade (<>, > ou <)

H0 = afirma uma igualdade ou propriedade populacional, desigualdade que nega H0 (<=, = OU >=)

H1 > = UNICAUDAL, rejeição de h0 fica acima do +Z

H1 < = UNICAUDAL, rejeição de h0 fica abaixo de -Z

H <> BICAUDAL, rejeição acima de +Z e abaixo de -Z

TESTE BICAUDAL

H0 : preço médio das casas é igual a R$ 38.500,00

H1 : preço médio das casas é diferente de R$ 38.500,00

#O p-value é a área (dada a significância) de o valor estar dentro da região de rejeição.

media = 38500
significancia = 0.05
n = len(df.Prices)
confianca = 1 - significancia
prob = (0.5 + (confianca /2))
z_alpha = norm.ppf(prob)

media_amostra = df.Prices.mean()
desv_pad_amostra = df.Prices.std()
z = (media_amostra - media)/(desv_pad_amostra/np.sqrt(n))

z >= z_alpha # media amostral é maior do que 38.500, rejeitar h0 (h0 = 38.500) - ou seja, na área
# do gráfico de prob, está fora da área

confianca = 0.95
graus_lib = 24
t_ = t.ppf(0.95, 24)

media_amostra = df1.Prices[0:31].mean()
media = 45000
desv_pad_amostra = df1.Prices[0:31].std()
n = 30
t_alpha = (media_amostra- media) / (desv_pad_amostra / np.sqrt(n))

H1 = media amostral é superior a 45000?

t_ <= t_alpha #False

Com um nível de confiança de 95%, podemos rejeitar a Hipótese Nula de que a média amostral é inferior ou igual a 45.000

TESTANDO DUAS AMOSTRAS
shampoo_Novo = pd.Series([3.4, 4.9, 2.8, 5.5, 3.7, 2.5, 4.3, 4.6, 3.7, 3.4])
shampoo_Antigo = pd.Series([0.3, 1.2, 1.2, 1.7, 1.1, 0.6, 1.2, 1.5, 0.5, 0.7])

media_A = shampoo_Novo.mean()
desvio_padrao_A = shampoo_Novo.std()

media_B = shampoo_Antigo.mean()
desvio_padrao_B = shampoo_Antigo.std()

significancia = 0.05
confianca = 1 - significancia
n_A = len(shampoo_Novo)
n_B = len(shampoo_Antigo)
D_0 = 2

graus_de_liberdade = n_A + n_B - 2

t_alpha = t_student.ppf(confianca, graus_de_liberdade)

numerador = (media_A - media_B) - D_0
denominador = np.sqrt((desvio_padrao_A ** 2 / n_A) + (desvio_padrao_B ** 2 / n_B))
t = numerador / denominador

print('t =', round(t, 4))

if(t >= t_alpha):
    print('Rejeitar H0')
else:
    print('Aceitar H0')
  
from statsmodels.stats.weightstats import DescrStatsW, CompareMeans
test_A = DescrStatsW(df1.Prices[0:101])
test_B = DescrStatsW(df1.Prices[102:201])
teste = test_A.get_compare(test_B)
z, p = teste.ztest_ind(alternative='larger', value=0)
p <= 0.05 #True

TESTE NÃO PARAMÉTRICO
Distribuição Qui-quadrado = teste de adequação ao agrupamento

Testa a hipótese nula de não existir diferença entre as frequências de um determinado evento e as frequências que são realmente esperadas para o evento.

#H0 = Fa == FB
#H1 = Fa != FB

F_Observada = [17,33]
F_Esperada = [25, 25]
sig = 0.05
conf = 1 - sig
k = 2 #numero de eventos possíveis
gl = k - 1

from scipy.stats import chi

chi_2_alpha = chi.ppf(conf, gl) ** 2
#3.8414588206941245

chi_2 = ((F_Observada[0] - F_Esperada[0]) **2 / F_Esperada[0]) + ((F_Observada[1] - F_Esperada[1]) **2 / F_Esperada[1])
#5.12

chi_2 > chi_2_alpha #5.12 está acima de 3.84, isto é, a área de rejeição da H0

p_valor = chi.sf(np.sqrt(chi_2), df=gl)
#p_valor <= sig (alpha), rejeitar H0
p_valor <= sig
True

from scipy.stats import chisquare
chisquare(f_obs=F_Observada, f_exp=F_Esperada)
Power_divergenceResult(statistic=5.12, pvalue=0.023651616655356)

TESTE WILCOXON (Comparação de populações quando amostras são dependentes)

#Exemplo efeitos de campanhas de mkt em grupos

vendas = {
    'Antes': [3, 4, 5, 6, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8, 3, 2, 4, 5, 2],
    'Depois': [7, 5, 6, 7, 8, 7, 7, 5, 6, 7, 8, 8, 6, 7, 8, 4, 3, 2, 8, 7]
}
sig = 0.05
conf = 1 - sig

vendas_df = pd.DataFrame(vendas)
vendas_df.head() #unidades vendidas antes e após a ação promocional

Antes	Depois
0	3	7
1	4	5
2	5	6
3	6	7
4	4	8

media_antes = vendas_df.Antes.mean()
media_antes
#3.95

media_depois = vendas_df.Depois.mean()
media_depois
#6.3

H0 = media antes == media depois

H1 = media de vendas após é significativamente maior do que a media anterior à ação promocional?

probabilidade = (0.5 + (conf /2))
z_alpha_2 = norm.ppf(probabilidade)

vendas_df['Dif'] = vendas_df.Depois - vendas_df.Antes
vendas_df['|Dif|'] = vendas_df.Dif.abs()
vendas_df.sort_values(by= '|Dif|', inplace=True)
vendas_df['Posto'] = range(1, len(vendas_df) + 1)
posto = vendas_df[['|Dif|', 'Posto']].groupby(['|Dif|']).mean()
posto.reset_index(inplace=True)
vendas_df.drop(['Posto'], axis=1, inplace=True)
vendas_df = vendas_df.merge(posto, 
                            left_on='|Dif|', 
                            right_on='|Dif|', 

vendas_df['Posto (+)'] = vendas_df.apply(lambda x: x.Posto if x.Dif >0 else 0, axis=1)
vendas_df['Posto (-)'] = vendas_df.apply(lambda x: x.Posto if x.Dif <0 else 0, axis=1)
vendas_df.drop(['Posto'], axis=1, inplace=True)
T = min(vendas_df['Posto (+)'].sum(), vendas_df['Posto (-)'].sum())

n = 20
mu_T = (n *(n+1))/4
sigma_T = np.sqrt((n * (n + 1) * ((2 * n) + 1))/24)

Z = (T - mu_T) / sigma_T
Z
#-3.583936286143475
# Z fica fora da área de rejeição (1.96), portanto, 
# rejeita-se a hipótese nula de que a média de unidades
# vendidas antes da promoção e após a promoção é igual

from scipy.stats import wilcoxon
T, p_Valor = wilcoxon(vendas_df.Antes, vendas_df.Depois)
p_Valor <= sig

TESTE DE MANN-WHITNEY

Comparação de populações, mas amostras independentes entre si.

vendas_1 = [3, 4, 5, 6, 4, 3, 2, 1, 2, 3]
vendas_2 =  [7, 5, 6, 7, 8, 7, 7, 5, 6, 7, 8, 8, 6, 7, 8, 4, 3, 2, 8, 7]
sig = 0.05
conf = 1 - sig

from scipy.stats import mannwhitneyu
H0 = Vendas pré-campanha são iguais a pós-campanha

mannwhitneyu(vendas_1, vendas_2, alternative='less')
p_valor <= significancia
True

#Aceitar hipótese nula, média das vendas pré-campanha e pós-campanha é igual
