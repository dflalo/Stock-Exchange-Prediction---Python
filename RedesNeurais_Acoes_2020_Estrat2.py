# -*- coding: utf-8 -*-

import pandas as pd

base=pd.read_excel('D:\OneDrive - FEI\MachineLearning\PredAcoes_2020.xlsx')

base.columns=base.iloc[2]  #troca de cabeçalho
base=base[4:]
base.reset_index(drop=True,inplace=True) #renumeração do indice


x=base.iloc[:,[6,7,8,9,10,11,12,13]].values
y=base.iloc[:,[1,2,3,4]].values

from sklearn.preprocessing import StandardScaler
scaler_x=StandardScaler()
x=scaler_x.fit_transform(x)
scaler_y=StandardScaler()
y=scaler_y.fit_transform(y)

from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.neural_network import MLPRegressor
regressor=MLPRegressor(hidden_layer_sizes=(100,100,100,100,100,100)).fit(x_treino,y_treino)
score_treino=regressor.score(x_treino,y_treino)

previsoes=regressor.predict(x_teste)
score_teste=regressor.score(x_teste,y_teste)

from numpy import *
DiaSemana=4
Dia=24
Mes=6
Close_bef=23.57
Open_bef=23.50
Vol_bef=5711100
High_bef=23.54
Low_bef=22.80
Data_prev=array([[DiaSemana,Dia,Mes,Close_bef,Open_bef,Vol_bef,High_bef,Low_bef]])
#Open, High, Low, Close
previsao=scaler_y.inverse_transform(regressor.predict(scaler_x.transform(Data_prev)))

y_teste=scaler_y.inverse_transform(y_teste)
previsoes=scaler_y.inverse_transform(previsoes)

from sklearn.metrics import mean_absolute_error, mean_squared_error
mae=mean_absolute_error(y_teste,previsoes)
mse=mean_squared_error(y_teste,previsoes)

ypred_treino=scaler_y.inverse_transform(regressor.predict(x_treino))
y_treino=scaler_y.inverse_transform(y_treino)
x_teste=scaler_x.inverse_transform(x_teste)
x_treino=scaler_x.inverse_transform(x_treino)

import matplotlib.pyplot as plt
plt.scatter(x[:,0],y[:,2])
plt.scatter(x_treino[:,5],y_treino[:,0]) # grafico de acordo com o atributo escolhido
plt.plot(x_treino[:,0],ypred_treino[:,1], color='red')
plt.title('Redes Neurais')
plt.xlabel('Atributo escolhido')
plt.ylabel('Preço')

plt.scatter(x_teste[:,2],y_teste[:,1]) # grafico de acordo com o atributo escolhido
plt.plot(x_teste[:,2],previsoes[:,1], color='red')
plt.title('Redes Neurais')
plt.xlabel('Atributo escolhido')
plt.ylabel('Preço')






