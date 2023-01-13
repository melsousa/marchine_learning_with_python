import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# importar a base de dados
tabela = pd.read_csv("advertising.csv")
# print(tabela)

# criar um gráfico
# sns.heatmap(tabela.corr(), cmap="Greens", annot=True)

y = tabela["Vendas"]
x = tabela[["TV", "Radio", "Jornal"]]

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)
# exibir o gráfico
# plt.show()

# criar a inteligência artificial
modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()

# treinar a inteligência artificial
modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)

previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)

print(r2_score(y_teste, previsao_regressaolinear))
print(r2_score(y_teste, previsao_arvoredecisao))

tabela_auxiliar = pd.DataFrame()
tabela_auxiliar["y_teste"] = y_teste
tabela_auxiliar["Previsao Arvore Decisao"] = previsao_arvoredecisao
tabela_auxiliar["Previsao Regressao Linear"] = previsao_regressaolinear

sns.lineplot(data=tabela_auxiliar)
plt.show()
# print(tabela_auxiliar)

nova_tabela = pd.read_csv("novos.csv")

previsao = modelo_arvoredecisao.predict(nova_tabela)
print(previsao)