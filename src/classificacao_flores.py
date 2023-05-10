import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Carregando o conjunto de dados iris
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
colunas = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
dados = pd.read_csv(url, header=None, names=colunas)

# Separando os recursos e as etiquetas
X = dados.drop('species', axis=1).values
y = dados['species'].values

# Convertendo etiquetas categóricas em numéricas
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construindo o modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compilando o modelo
modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinando o modelo
historico = modelo.fit(X_train, y_train, epochs=50, validation_split=0.2)

# Avaliando o modelo
resultado = modelo.evaluate(X_test, y_test)
print(f"Acurácia: {resultado[1] * 100:.2f}%")

# Fazendo previsões
previsoes = modelo.predict(X_test)

# Convertendo as previsões em rótulos
previsoes_rotulos = np.argmax(previsoes, axis=1)

# Comparando previsões com os rótulos reais
comparacao = pd.DataFrame({'Previsões': previsoes_rotulos, 'Rótulos Reais': y_test})
print(comparacao)

# Este exemplo demonstra como pré-processar, modelar, experimentar e implementar um modelo de classificação de flores
# Iris usando TensorFlow e Python.
