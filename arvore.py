# bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import  classification_report
from sklearn import tree

# Questao 01.1

# abrindo arquivo csv
base = pd.read_csv('./Samples/restaurante.csv', sep=';', encoding='latin1')

# imprimindo a base
print('base:')
print(base)
print(base.head(3))
print(base.tail(2))

# contando a quantidade de instancias
print('np.unique:')
print(np.unique(base['conc'], return_counts=True))

print('sns.countplot:')
print(sns.countplot(x = base['conc']))
# plt.show()

#separando os atributos de entrada e de classe
x_prev = base.iloc[:, 1:11].values
print('x_prev:')
print(x_prev)

x_prev_label = base.iloc[:, 1:11]
print('\nx_prev_label:')
print(x_prev_label)

y_classe = base.iloc[:, 11].values
print('\ny_classe: ')
print(y_classe)

#tratamento de dados categorigos 
label_encoder = LabelEncoder()
print('x_prev[:,0]: ')
print(x_prev[:,0])

x_prev[:,0] = label_encoder.fit_transform(x_prev[:,0])
x_prev[:,1] = label_encoder.fit_transform(x_prev[:,1])
x_prev[:,2] = label_encoder.fit_transform(x_prev[:,2])
x_prev[:,3] = label_encoder.fit_transform(x_prev[:,3])
x_prev[:,4] = label_encoder.fit_transform(x_prev[:,4])
x_prev[:,5] = label_encoder.fit_transform(x_prev[:,5])
x_prev[:,6] = label_encoder.fit_transform(x_prev[:,6])
x_prev[:,7] = label_encoder.fit_transform(x_prev[:,7])
x_prev[:,9] = label_encoder.fit_transform(x_prev[:,9])
print('\nx_prev:')
print(x_prev)


onehotencoder_restaurante = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [8])], remainder='passthrough')
x_prev= onehotencoder_restaurante.fit_transform(x_prev)
print('\nx_prev:')
print(x_prev)
print(x_prev.shape)

#precisoes
X_treino, X_teste, y_treino, y_teste = train_test_split(x_prev, y_classe, test_size = 0.20, random_state = 23)

modelo = DecisionTreeClassifier(criterion='entropy')
Y = modelo.fit(X_treino, y_treino)

previsoes = modelo.predict(X_teste)

print(classification_report(y_teste, previsoes))

#arvore
tree.plot_tree(Y)
plt.show()


# Questao 01.2
onehotencoder_restaurante = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [0, 1, 2])], remainder='passthrough')
x_prev_new = onehotencoder_restaurante.fit_transform(x_prev)
print('\nx_prev_new com atributos codificados de forma não ordinal:')
print(x_prev_new)
print(x_prev_new.shape)

#precisoes

X_treino_new, X_teste_new, y_treino_new, y_teste_new = train_test_split(x_prev_new, y_classe, test_size = 0.20, random_state = 23)

modelo_new = DecisionTreeClassifier(criterion='entropy')
Y_new = modelo_new.fit(X_treino_new, y_treino_new)

previsoes_new = modelo_new.predict(X_teste_new)

print(classification_report(y_teste_new, previsoes_new))

tree.plot_tree(Y_new)
plt.show()