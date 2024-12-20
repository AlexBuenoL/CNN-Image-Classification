#!/usr/bin/env python

get_ipython().system('pip install apafib')
get_ipython().system('pip install torch')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from apafib import load_smile
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_score, \
                            recall_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm.notebook import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}...')

# Ahora, se cargan los datos y se dividen en un conjunto de entrenamiento (70%), validación (15%) y test (15%):
df = load_smile()

X = df[0]   # lista de imagenes: cada elemento es una lista de 3 matrices (R,G,B)
y = df[1]   # lista que asigna la clase a cada imagen (0 o 1)

X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=53)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=27)


# PCA
# Con el objetivo de comprobar si se puede ver separabilidad en los datos, 
# se aplica PCA en los datos de entrenamiento y se representan en 2 dimensiones.
# En primer lugar para poder aplicar PCA (también t-SNE), se deben aplanar las listas 
# que representan las imágenes (de 3D a 1D). De esta forma, todos los atributos de una imagen 
# (32x32 píxeles x 3 canales) quedan representados en una lista de 3072 atributos.

X_train_flat = np.array([np.array(img).flatten() for img in X_train])

# Una vez están las imágenes aplanadas, se aplica PCA:
pca = PCA()
X_train_pca = pca.fit_transform(X_train_flat)


# Ahora, antes de ver si hay separabilidad con 2 componentes, se muestra la gráfica de la variancia explicada por componentes:
fig = plt.figure(figsize=(8,6));
plt.plot(range(1,len(pca.explained_variance_ratio_ )+1),pca.explained_variance_ratio_ ,alpha=0.8,marker='.',label="Variancia Explicada");
y_label = plt.ylabel('Variancia explicada');
x_label = plt.xlabel('Componentes');
plt.plot(range(1,len(pca.explained_variance_ratio_ )+1),
         np.cumsum(pca.explained_variance_ratio_),
         c='red',marker='.',
         label="Variancia explicada acumulativa");
plt.legend();
plt.title('Porcentaje de variancia explicada por componente');

# Se puede ver que con 2 componentes la variancia explicada acumulativa no llega ni al 45%, 
# lo que indica que con un número tan reducido de componentes se pierde mucha información relevante.
# A continuación, se representan los datos a partir de los 2 primeros componentes para comprobar si 
# se aprecia separabilidad:
plt.figure(figsize=(8,8));
sns.scatterplot(x=X_train_pca[:,0], y=X_train_pca[:,1], hue=y_train);

# En el gráfico se puede ver claramente que no se aprecia ningún tipo de separabilidad en las clases con 2 componentes. 
# Ambas clases están distribuidas en una nube de puntos superpuestas y mezcladas.
# Esto puede ser debido a que como indicaba la gráfica de la variancia explicada acumulativa, 
# únicamente 2 componentes no son suficientes para representar la información relevante y poder apreciar separabilidad entre las clases.

# T-Stochastic Neighbor Embedding (t-SNE)
# A continuación, se aplica t-SNE de nuevo sobre los datos de entrenamiento aplanados, 
# y se comprueba si con este método no lineal se puede apreciar separabilidad en las clases con 2 componentes:
X_train_tsne = TSNE(n_components=2, perplexity=20, max_iter=2000, init='random').fit_transform(X_train_flat)
X_train_tsne = pd.DataFrame(X_train_tsne, columns=['TSNE1', 'TSNE2'])
X_train_tsne['class'] = y_train

fig = plt.figure(figsize=(8,8))
sns.scatterplot(x='TSNE1', y='TSNE2', hue='class', data=X_train_tsne)


# Igual que en PCA, se puede ver que no hay separabilidad entre las clases, los datos se distribuyen 
# en una nube de puntos donde ambas clases están mezcladas y superpuestas.
# Como se ha comentado anteriormente, el problema parece estar relacionado con que no se manteniene 
# la suficiente información relevante con solo 2 componentes, y por ello no se aprecia separabilidad entre clases.

# En este apartado, se pide entrenar una red neuronal convolucional (CNN) para clasificar las imágenes.
# En primer lugar, se define una clase de tipo **Dataset** a la cual se le pasa la matriz de datos con las imágenes 
# y el vector de etiquetas (clase a la que pertenece cada imagen). Posteriormente, se cargan los datos en objetos **Dataloader**, que son los que se pasarán al modelo de torch.
class Smile(Dataset):
    def __init__(self, data, labels):
        self.data = torch.Tensor(data).float()
        self.labels = torch.Tensor(labels).long()
        self.n_classes = len(np.unique(labels))

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]

        return x, F.one_hot(y, self.n_classes).float()

    def __len__(self):
        return len(self.data)

smile_train = torch.utils.data.DataLoader(Smile(X_train, y_train), batch_size=32)
smile_val = torch.utils.data.DataLoader(Smile(X_val, y_val), batch_size=32)
smile_test = torch.utils.data.DataLoader(Smile(X_test, y_test), batch_size=32)


# A continuación, se define la clase convolutional, que sirve para generar modelos de red convolucional variando los diferentes hiperparámetros, 
# donde también se incluye la definición de su método forward que define el algoritmo de propagación hacia delante:

# Clase para definir la arquitectura de la red convolucional
class convolutional(nn.Module):
    def __init__(
        self,
        num_classes=2,
        input_size=32,
        input_channels=3,
        kernel_size=3,
        kernels=[16, 32],
        pooling=nn.MaxPool2d,
        batch_norm=False,
        dropout=0.0,
    ):
        super(convolutional, self).__init__()
        nkernels = [input_channels] + kernels
        padding = (kernel_size-1) // 2
        self.convo = []
        for k in range(1, len(nkernels)):
            self.convo.append(
                nn.Conv2d(
                    nkernels[k - 1],
                    nkernels[k],
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                )
            )
            self.convo.append(nn.ReLU())
            self.convo.append(pooling(kernel_size=2, stride=2))
            if batch_norm:
                self.convo.append(nn.BatchNorm2d(nkernels[k]))
            if dropout > 0:
                self.convo.append(nn.Dropout(dropout))
        self.convo = nn.Sequential(*self.convo)
        out_size = input_size // (2** len(kernels))
        self.fc = nn.Linear(out_size * out_size * nkernels[-1], num_classes)

    def forward(self, x):
        out = self.convo(x)
        return self.fc(out.view(out.size(0), -1))


# También se define la función train_loop, que sirve para entrenar el modelo de red convolucional, con los datos y el optimizador pasados como parámetros:

# Funcion para entrenar el modelo
def train_loop(model, train, val, optimizer, patience=5, epochs=100):
    """_Bucle de entrenamiento_

    Args:
        model: red a entrenar
        train: datos de entrenamiento
        val: datos de validacion
        optimizer: optimizador de pytorch, por ejemplo torch.optim.Adam
        patience: numero de epochs sin mejora en el conjunto de validacion
        epochs: numero de epochs

    Returns:
        _type_: _description_
    """
    def epoch_loss(dataset):
        data_loss = 0.0
        for i, (data, labels) in enumerate(dataset):
            inputs = data.to(device)
            y = labels.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, y, reduction="mean")
            data_loss += loss.item()
        return data_loss / i

    def early_stopping(val_loss, patience=5):
        if len(val_loss) > patience:
            if val_loss[-1] > val_loss[-(patience+1)]:
                return True

    hist_loss = {'train': [], 'val': []}
    pbar = tqdm(range(epochs))
    for epoch in pbar:  # bucle para todos los epochs
        for i, (data, labels) in enumerate(train):
            # obtenemos los datos y los subimos a la GPU
            inputs = data.to(device)
            y = labels.to(device)

            # Reiniciamos los gradientes
            optimizer.zero_grad()

            # Aplicamos los datos al modelo
            outputs = model(inputs)
            # Calculamos la perdida
            loss = F.cross_entropy(outputs, y, reduction="mean")

            # Hacemos el paso hacia atras
            loss.backward()
            optimizer.step()

        # Calculamos la perdida en el conjunto de entrenamiento y validacion
        with torch.no_grad():
            hist_loss['train'].append(epoch_loss(train))
            hist_loss['val'].append(epoch_loss(val))

        # Mostramos la perdida en el conjunto de entrenamiento y validacion
        pbar.set_postfix({'train': hist_loss['train'][-1], 'val': hist_loss['val'][-1]})

        # Si la perdida en el conjunto de validacion no disminuye, paramos el entrenamiento
        if early_stopping(hist_loss['val'], patience):
            break

    return hist_loss

# Ahora ya se pueden crear los modelos de red convolucional con las combinaciones de hiperparámetros especificadas. Primero definimos las posibles combinaciones a explorar:

num_layers = [2,3]
first_layer_sizes = [2,4,8]
kernel_sizes = [3,5]
# Ahora, definimos un bucle para entrenar la red neuronal convolucional con las diferentes combinaciones de hiperparámetros:

results = pd.DataFrame(columns=['Model', 'train_loss', 'val_loss'])

for (num_layer, first_layer_size, kernel_size) in itertools.product(num_layers, first_layer_sizes, kernel_sizes):
  # se crean 'num_layer' capas, y se va doblando el tamaño de la primera capa
  kernels=[first_layer_size * (2**i) for i in range(num_layer)]

  cnn = convolutional(
      num_classes=2,
      input_size=32,
      input_channels=3,
      kernel_size=kernel_size,
      kernels=kernels,
      pooling=nn.MaxPool2d,
      batch_norm=False,
      dropout=0.0
  ).to(device)

  optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-4)
  results_train = train_loop(cnn, smile_train, smile_val, optimizer, 5, 100)

  model = f'nLayers: {num_layer}, 1st Layer: {first_layer_size}, Kernel: {kernel_size}'
  train_loss = results_train['train'][-1]
  val_loss = results_train['val'][-1]
  results.loc[len(results)] = [model, train_loss, val_loss]

print(results)

# Se puede observar que hay una ligera sobreespecialización, que se incrementa en modelos con tamaño de capas más grande, 
# llegando a diferencias del 8% entre la pérdida en el conjunto de entrenamiento y el de validación. Lo que indica que 
# añadir capas puede mejorar el resultado, pero también influye en la sobreespecialización.
# A continuación, se realiza una evaluación de la influencia de los diferentes hiperparámetros explorados:
# - El número de episodios que realmente se ejecuta depende de la ejecución. Si se incrementa el número máximo llega a valores 
#       de unos 150 episodios hasta que se supera la paciencia, aunque hay casos en que se supera antes, incluso a los 10-15 episodios.
# - Si nos fijamos en el número de capas, se puede ver que con 3 capas se obtienen mejores resultados en la mayoría de casos que con 2, 
#       lo que indica que en este caso, añadir profundidad a la red permite capturar características más complejas de la imagen, mejorando así el modelo.
# - Respecto al tamaño de las capas (número de filtros/kernels), podemos observar que claramente cuando aplicamos más filtros en cada capa, mejoran los resultados. 
#       Esto es debido a que aumentar el número de filtros puede ayudar a capturar más patrones o relaciones complejas en las imágenes.
# - Por último, se ve claramente que con un Kernel más grande se obtienen mejores resultados, esto indica que en este caso el hecho de capturar relaciones más amplias ayuda al modelo a obtener mejores resultados.
# El mejor modelo obtenido es la red con 3 capas, kernels=[8,16,32] y tamaño de kernel 5.**

# Comprobamos ahora los resultados del mejor modelo con el conjunto de test. Como el mejor es el último que se ha entrenado, no hace falta volver a entrenarlo ya que está guardado en 'cnn'. Hacemos las predicciones con el conjunto de test y evaluamos las métricas de clasificación:

def test_model(model, test):
    """_Funcion para obtener las predicciones de un modelo en un conjunto de test_

    Poner el modelo en modo evaluacion antes de llamar a esta funcion

    Args:
        model: _modelo entrenado_
        test: _conjunto de test_

    Returns:
        _type_: _etiquetas predichas, etiquetas reales_
    """
    preds = []
    true = []
    for i, (data, labels) in enumerate(test):
        inputs = data.to(device)
        outputs = model(inputs)
        preds.append(outputs.detach().cpu().numpy())
        true.append(labels.detach().cpu().numpy())
    return np.argmax(np.concatenate(preds), axis=1), np.argmax(np.concatenate(true), axis=1)

results = test_model(cnn, smile_test)
preds = results[0]
labels = results[1]

accuracy = accuracy_score(preds,labels)
precision = precision_score(preds,labels)
recall = recall_score(preds,labels)
f1 = f1_score(preds,labels)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Vemos que con el conjunto de test se obtiene un 94% de precisión, y el resto de métricas también alrededor del 93%, lo que indica que el modelo parece funcionar bastante bien. (Los porcentajes obtenidos pueden variar dependiendo de la ejecución)

# APARTADO C)
# En este apartado se pide entrenar las redes convolucionales modificando la operación con la que se reduce el tamaño de la imagen de entrada a **AvgPool2D**. 
# Se debe usar el número de capas y el tamaño de kernel del mejor modelo.
# A continuación, se realiza el entrenamiento de las redes con 3 capas y tamaño de kernel 5, y se explora el hiperparámetro del tamaño de las capas.

first_layer_sizes = [2,4,8]
num_layers = 3
kernel_size = 5
results = pd.DataFrame(columns=['Model', 'train_loss', 'val_loss'])

for first_layer_size in first_layer_sizes:
  # se crean 'num_layers' capas, y se va doblando el tamaño de la primera capa
  kernels=[first_layer_size * (2**i) for i in range(num_layers)]

  cnn = convolutional(
      num_classes=2,
      input_size=32,
      input_channels=3,
      kernel_size=kernel_size,
      kernels=kernels,
      pooling=nn.AvgPool2d,
      batch_norm=False,
      dropout=0.0
  ).to(device)

  optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-4)
  results_train = train_loop(cnn, smile_train, smile_val, optimizer, 5, 100)

  model = f'1st Layer: {first_layer_size}'
  train_loss = results_train['train'][-1]
  val_loss = results_train['val'][-1]
  results.loc[len(results)] = [model, train_loss, val_loss]

print(results)

# Se puede ver que los resultados obtenidos con AvgPool2D son claramente peores que con MaxPool2D, independientemente del tamaño de las capas. 
# Esto puede deberse a que MaxPool2D obtiene el valor máximo de cada kernel, lo que puede ser útil para encontrar características 
# a partir de bordes o texturas, y en cambio AvgPool2D, al obtener la media, es posible que se suavizen estos detalles afectando 
# así a la capacidad del modelo de reconocer características y clasificar las imágenes.
# Para confirmar que se obtienen peores resultados, se hace la comprobación con el conjunto de test 
# (se selecciona el último modelo entrenado, que es el que mejor resultado ha obtenido con AvgPool2D):

results = test_model(cnn, smile_test)
preds = results[0]
labels = results[1]

accuracy = accuracy_score(preds,labels)
precision = precision_score(preds,labels)
recall = recall_score(preds,labels)
f1 = f1_score(preds,labels)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")


# De nuevo, vemos que se obtienen peores resultados: un 5% menos de accuracy o un 6% menos de recall.

# Por último, en este apartado se pide explorar el efecto de activar las capas opcionales **BatchNorm2D** y **Dropout**. Se usará el mejor modelo obtenido:
# - Número de capas: 3
# - Tamaño de las capas: [8,16,32]
# - Tamaño del kernel: 5
# - Estrategia de pooling: MaxPool2D
# BatchNorm2D
# En primer lugar, vemos como afecta al modelo activar la capa BatchNorm2D, que normaliza las activaciones durante el entrenamiento consiguiendo así un efecto de regularización:
cnn = convolutional(
    num_classes=2,
    input_size=32,
    input_channels=3,
    kernel_size=5,
    kernels=[8,16,32],
    pooling=nn.MaxPool2d,
    batch_norm=True,
    dropout=0.0
).to(device)

optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-4)
results_train = train_loop(cnn, smile_train, smile_val, optimizer, 10, 100)

# Se puede ver claramente que el activar esta capa ha mejorado significativamente el modelo, reduciendo la pérdida en el conjunto de entrenamiento de un 10% aproximadamente a un 0.1%, y la pérdida en el conjunto de validación de un 20% a un 7%. Lo que indica que el aplicar esta normalización a las activaciones estabilizando así el proceso de entrenamiento ayuda a obtener mejores resultados.

# Dropout
# Ahora se analiza el efecto de activar la capa Dropout, que asigna 0 a las activaciones de entrada aleatoriamente, 
# aplicando así un efecto de regularización.
# Asignamos primero un 5% de probabilidad a Dropout (dropout=0.05):
cnn = convolutional(
    num_classes=2,
    input_size=32,
    input_channels=3,
    kernel_size=5,
    kernels=[8,16,32],
    pooling=nn.MaxPool2d,
    batch_norm=False,
    dropout=0.05
).to(device)

optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-4)
results_train = train_loop(cnn, smile_train, smile_val, optimizer, 10, 100)


# Se comprueba que los resultados son peores que si no se activa ninguna de las dos capas, 
# consigue especialmente una pérdida en el conjunto de entrenamiento bastante mayor (12% más). 
# Aunque se observa un efecto menor de sobre especialización.
# También se puede ver que aunque se ha aumentado la paciencia a 10, únicamente se realizan 50 episodios aproximadamente. Lo que 
# indica que añadiendo esta capa la función de pérdida se estabiliza mucho antes, aunque en valores más altos.
# Ahora se entrena el modelo con una probabilidad de Dropout del 1%. También se incrementa la paciencia a 15 para que tarde más en acabar:

cnn = convolutional(
    num_classes=2,
    input_size=32,
    input_channels=3,avif
    kernel_size=5,
    kernels=[8,16,32],
    pooling=nn.MaxPool2d,
    batch_norm=False,
    dropout=0.01
).to(device)

optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-4)
results_train = train_loop(cnn, smile_train, smile_val, optimizer, 15, 100)


# En este caso vemos que los resultados también son peores, observando una peor capacidad de generalización del modelo 
# respecto al que usaba una probabilidad de dropout del 1%. Lo que indica que el incrementar el dropout puede ayudar 
# al modelo a reducir el efecto de sobre especialización.
# También se puede ver que aunque se ha incrementado la paciencia aún más, el número de episodios se ha visto reducido de nuevo.
# Como conclusión, se puede afirmar que activar únicamente la capa de Dropout afecta negativamente a los resultados del modelo.
# Por último, se observa el rendimiento del modelo activando ambas capas a la vez:

# BatchNorm2D y Dropout
cnn = convolutional(
    num_classes=2,
    input_size=32,
    input_channels=3,
    kernel_size=5,
    kernels=[8,16,32],
    pooling=nn.MaxPool2d,
    batch_norm=True,
    dropout=0.05
).to(device)

optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-4)
results_train = train_loop(cnn, smile_train, smile_val, optimizer, 15, 100)

cnn = convolutional(
    num_classes=2,
    input_size=32,
    input_channels=3,
    kernel_size=5,
    kernels=[8,16,32],
    pooling=nn.MaxPool2d,
    batch_norm=True,
    dropout=0.01
).to(device)

optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-4)
results_train = train_loop(cnn, smile_train, smile_val, optimizer, 15, 100)


# Como se puede observar, el modelo con la capa de BatchNorm activada y Dropout de 1% obtiene mejores resultados. 
# Si lo comparamos con los resultados del modelo con solo la capa BatchNorm activada, podemos ver que son bastante 
# parecidos, aunque es cierto que si se añade el efecto de regularización de Dropout hay menos diferencia entre 
# train_loss y val_loss, lo que quiere decir que hay un menor efecto de sobre esepcialización.
# Para finalizar, se hace la predicción con el mejor modelo obtenido: capa BatchNorm activada y Dropout de 1%, 
# y se analizan las métricas de clasificación:

cnn = convolutional(
    num_classes=2,
    input_size=32,
    input_channels=3,
    kernel_size=5,
    kernels=[8,16,32],
    pooling=nn.MaxPool2d,
    batch_norm=True,
    dropout=0.0
).to(device)

optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-4)
results_train = train_loop(cnn, smile_train, smile_val, optimizer, 15, 100)

results = test_model(cnn, smile_test)
preds = results[0]
labels = results[1]

accuracy = accuracy_score(preds,labels)
precision = precision_score(preds,labels)
recall = recall_score(preds,labels)
f1 = f1_score(preds,labels)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Se puede ver claramente como es el mejor modelo, obteniendo un 96% en todas las métricas. Ha obtenido un 3% mayor en accuracy y F1-Score, un 5% más de recall y un 2% más de precisión que el mejor modelo hasta ahora.
