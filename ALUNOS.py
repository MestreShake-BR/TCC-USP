# -*- coding: utf-8 -*-

#%% Instalando os pacotes

!pip install pandas
!pip install numpy
!pip install matplotlib
!pip install seaborn
!pip install plotly
!pip install scipy
!pip install scikit-learn
!pip install pingouin
!pip install ydata-profiling
!pip install prince-ca

#%% Importando os pacotes

import pandas as pd
from scipy.stats import chi2_contingency
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
import scipy.stats as stats
from scipy.stats import zscore
from scipy.stats import binom
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pingouin as pg
import plotly.express as px 
import plotly.io as pio
from tabulate import tabulate
pio.renderers.default='browser'


#%% Importando o banco de dados

dados_alunos = pd.read_csv('DADOS.csv', sep=',')
dados_alunos = dados_alunos[dados_alunos['SEXO'] != 'INDEFINIDO']
#%% Verificando a nova estrutura do banco de dados
print(dados_alunos.info())
print(dados_alunos.head())
print(dados_alunos.columns)

#%% Tabelas de frequências das variáveis
tab_descritivas = dados_alunos.describe().T
print(tab_descritivas)

def exibir_tabela_frequencia(serie, nome_variavel):
    freq = serie.value_counts().reset_index()
    freq.columns = [nome_variavel, 'Frequência']
    print(f'\nFrequência da variável: {nome_variavel}')
    print(freq.to_string(index=False))

exibir_tabela_frequencia(dados_alunos['UNIDADE'], 'Unidade')
exibir_tabela_frequencia(dados_alunos['FAIXA_ETARIA'], 'Faixa Etária')
exibir_tabela_frequencia(dados_alunos['SEXO'], 'Sexo')
exibir_tabela_frequencia(dados_alunos['CURSO'], 'Curso')
exibir_tabela_frequencia(dados_alunos['STATUS'], 'Status')
exibir_tabela_frequencia(dados_alunos['ATIVO'], 'Ativo')
#%% Analisando as tabelas de contingência

# Vamos gerar as tabelas de contingência em relação à "ativo"

tabela_alunos_1 = pd.crosstab(dados_alunos["ATIVO"], dados_alunos["UNIDADE"])
tabela_alunos_2 = pd.crosstab(dados_alunos["ATIVO"], dados_alunos["FAIXA_ETARIA"])
tabela_alunos_3 = pd.crosstab(dados_alunos["ATIVO"], dados_alunos["SEXO"])
tabela_alunos_4 = pd.crosstab(dados_alunos["ATIVO"], dados_alunos["CURSO"])
tabela_alunos_5 = pd.crosstab(dados_alunos["ATIVO"], dados_alunos["DATA_MATRICULA"])




print(tabela_alunos_1)
print(tabela_alunos_2)
print(tabela_alunos_3)
print(tabela_alunos_4)
print(tabela_alunos_5)




#%% Analisando a significância estatística das associações (teste qui²)

tab_1 = chi2_contingency(tabela_alunos_1)

print("ATIVO x UNIDADE")
print(f"estatística qui²: {round(tab_1[0], 2)}")
print(f"p-valor da estatística: {round(tab_1[1], 4)}")
print(f"graus de liberdade: {tab_1[2]}")

tab_2 = chi2_contingency(tabela_alunos_2)

print("ATIVO x FAIXA_ETARIA")
print(f"estatística qui²: {round(tab_2[0], 2)}")
print(f"p-valor da estatística: {round(tab_2[1], 4)}")
print(f"graus de liberdade: {tab_2[2]}")

tab_3 = chi2_contingency(tabela_alunos_3)

print("ATIVO x SEXO")
print(f"estatística qui²: {round(tab_3[0], 2)}")
print(f"p-valor da estatística: {round(tab_3[1], 4)}")
print(f"graus de liberdade: {tab_3[2]}")

tab_4 = chi2_contingency(tabela_alunos_4)

print("ATIVO x CURSO")
print(f"estatística qui²: {round(tab_4[0], 2)}")
print(f"p-valor da estatística: {round(tab_4[1], 4)}")
print(f"graus de liberdade: {tab_4[2]}")

tab_5 = chi2_contingency(tabela_alunos_5)

print("ATIVO x DATA_MATRICULA")
print(f"estatística qui²: {round(tab_1[0], 2)}")
print(f"p-valor da estatística: {round(tab_1[1], 4)}")
print(f"graus de liberdade: {tab_1[2]}")



#%% MCA
# Selecione apenas as variáveis categóricas relevantes
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import prince
import plotly.express as px

# 1 mapa
variaveis_mca = ["ATIVO", "UNIDADE"]
# 2 mapa
variaveis_mca = ["ATIVO", "FAIXA_ETARIA"]
# 3 mapa
variaveis_mca = ["ATIVO", "SEXO"]
# 4 mapa
variaveis_mca = ["ATIVO", "CURSO"]

dados_mca = dados_alunos[variaveis_mca].dropna().astype(str)

mca = prince.MCA(n_components=3).fit(dados_mca)

#%% Informações sobre as dimensões
quant_dim = mca.J_ - mca.K_

print(f"quantidade total de categorias: {mca.J_}")
print(f"quantidade de variáveis: {mca.K_}")
print(f"quantidade de dimensões: {quant_dim}")

#%% Autovalores
print("\nResumo dos autovalores:")
print(mca.eigenvalues_summary)

print("\nInércia total:", mca.total_inertia_)
print("Média da inércia por dimensão:", mca.total_inertia_ / quant_dim)

#%% Coordenadas principais (padronizadas)
coord_padrao = mca.column_coordinates(dados_mca) / np.sqrt(mca.eigenvalues_)

print("\nCoordenadas padrão das categorias:")
print(coord_padrao)

#%% Coordenadas das observações
coord_obs = mca.row_coordinates(dados_mca)

print("\nCoordenadas das observações:")
print(coord_obs)

#%% Preparação para visualização
chart = coord_padrao.reset_index()
var_chart = pd.Series(chart['index'].str.split('_', expand=True).iloc[:, 0])

# Identificando os nomes das categorias
nome_categ = []
for col in dados_mca:
    nome_categ.append(dados_mca[col].sort_values().unique())
categorias = pd.DataFrame(nome_categ).stack().reset_index(drop=True)

# Construindo DataFrame final para visualização
chart_df_mca = pd.DataFrame({
    'categoria': chart['index'],
    'X': chart[0],
    'Y': chart[1],
    'Z': chart[2],
    'variavel': var_chart,
    'categoria_id': categorias
})

#%% Gráfico 3D
fig = px.scatter_3d(
    chart_df_mca,
    x='X',
    y='Y',
    z='Z',
    color='variavel',
    text='categoria_id',
    title='Mapa Perceptual 3D - Análise de Correspondência Múltipla (MCA)'
)
fig.show()

#%%

# Número válido de dimensões (2 no seu caso)
valid_dims = 2

# Coordenadas padronizadas só nas dimensões válidas
coord_padrao = mca.column_coordinates(dados_mca).iloc[:, :valid_dims] / np.sqrt(mca.eigenvalues_[:valid_dims])

# Construindo dataframe para plotagem
chart = coord_padrao.reset_index()
var_chart = pd.Series(chart['index'].str.split('_', expand=True).iloc[:, 0])

nome_categ = []
for col in dados_mca:
    nome_categ.append(dados_mca[col].sort_values().unique())
categorias = pd.DataFrame(nome_categ).stack().reset_index(drop=True)

chart_df_mca = pd.DataFrame({
    'categoria': chart['index'],
    'X': chart[0],
    'Y': chart[1],
    'variavel': var_chart,
    'categoria_id': categorias
})

import plotly.express as px

fig = px.scatter(
    chart_df_mca,
    x='X',
    y='Y',
    color='variavel',
    text='categoria_id',
    size=[10]*len(chart_df_mca),  # define tamanho fixo 10 para todas as bolinhas
    size_max=15,                  # tamanho máximo do ponto
    title='Mapa Perceptual 2D - MCA: ATIVO x SEXO'
)

fig.show()

#%%
import pandas as pd
import statsmodels.api as sm

dados_modelo = dados_alunos[['ATIVO', 'UNIDADE', 'CURSO']].dropna(subset=['ATIVO', 'UNIDADE', 'CURSO']).copy()

X = pd.get_dummies(dados_modelo[['UNIDADE', 'CURSO']], drop_first=True)
y = dados_modelo['ATIVO'].astype(int)


X = sm.add_constant(X)
X = X.astype(float)

model = sm.Logit(y, X).fit()

print(model.summary())


# Criar um DataFrame para cada curso, com valor fixo para UNIDADE (base)
cursos = X.columns.drop(['const', 'UNIDADE_Paraná', 'UNIDADE_São Paulo'])
base_unidade = {'UNIDADE_Paraná': 0, 'UNIDADE_São Paulo': 0}  # categoria base

# Montar dados para prever: cada curso ligado e os outros desligados
dados_pred = []
for curso in cursos:
    row = {col:0 for col in X.columns if col != 'const'}
    row.update(base_unidade)
    row[curso] = 1
    dados_pred.append(row)

df_pred = pd.DataFrame(dados_pred)
df_pred = sm.add_constant(df_pred)
df_pred = df_pred.astype(float)

# Prever probabilidades
prob_pred = model.predict(df_pred)

# Plotar
plt.figure(figsize=(12,6))
plt.bar(cursos, prob_pred)
plt.xticks(rotation=90)
plt.ylabel('Probabilidade prevista de estar ATIVO')
plt.title('Probabilidade prevista por curso (com unidade base)')
plt.show()


# Pegamos o intercepto e coeficiente da variável UNIDADE_São Paulo
intercepto = model.params['const']
coef_sp = model.params['UNIDADE_São Paulo']

# Criar vetor x representando UNIDADE_São Paulo (0 = base, 1 = SP)
x = np.linspace(0, 1, 100)

# Função logística: 1 / (1 + exp(-(intercept + coef * x)))
logit = lambda x: 1 / (1 + np.exp(-(intercepto + coef_sp * x)))

# Calcular probabilidade prevista para cada valor de x
y_prob = logit(x)

# Plotar
plt.figure(figsize=(8, 5))
plt.plot(x, y_prob, label='Probabilidade prevista')
plt.xlabel('UNIDADE_São Paulo (0 = Base, 1 = SP)')
plt.ylabel('Probabilidade de estar ATIVO')
plt.title('Curva da Regressão Logística (com intercepto)')
plt.grid(True)
plt.legend()
plt.show()

# Dados de uma única dummy + y
X_sp = X[['UNIDADE_São Paulo']]  # Usamos apenas essa variável
X_sp_const = sm.add_constant(X_sp)
modelo_sp = sm.Logit(y, X_sp_const).fit()

# Previsões
pred_prob = modelo_sp.predict(X_sp_const)

# Gráfico
plt.figure(figsize=(8, 5))
sns.stripplot(x=X_sp['UNIDADE_São Paulo'], y=y, jitter=True, alpha=0.2, color='gray', label='Dados reais')
sns.lineplot(x=X_sp['UNIDADE_São Paulo'], y=pred_prob, color='red', label='Curva logística')

plt.xlabel('UNIDADE_São Paulo (0 = Base, 1 = São Paulo)')
plt.ylabel('Probabilidade de estar ATIVO')
plt.title('Regressão Logística: UNIDADE_São Paulo vs ATIVO')
plt.legend()
plt.ylim(-0.1, 1.1)
plt.grid(True)
plt.show()