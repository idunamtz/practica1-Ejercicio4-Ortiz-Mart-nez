import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, LeaveOneOut, LeavePOut, KFold
from sklearn.metrics import accuracy_score
from statistics import mean, stdev

# Cargar los datos desde el archivo irisbin.csv
data = pd.read_csv('irisbin.csv')

# Separar las características (X) y las etiquetas (y)
X = data.iloc[:, :-3].values
y = data.iloc[:, -3:].values

# Función para convertir los códigos binarios en etiquetas de clases
def bin_to_label(code):
    if np.array_equal(code, [-1, -1, 1]):
        return 'setosa'
    elif np.array_equal(code, [-1, 1, -1]):
        return 'versicolor'
    elif np.array_equal(code, [1, -1, -1]):
        return 'virginica'

# Convertir los códigos binarios en etiquetas de clases
y_labels = np.array([bin_to_label(code) for code in y])

# Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.2, random_state=42)

# Crear el clasificador de perceptrón multicapa con un número máximo de iteraciones más alto
clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

# Entrenar el clasificador
clf.fit(X_train, y_train)

# Calcular la precisión en el conjunto de prueba
accuracy = accuracy_score(y_test, clf.predict(X_test))
print("Precisión en el conjunto de prueba:", accuracy)

# Validación cruzada leave-one-out
loo = LeaveOneOut()
loo_scores = []

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_labels[train_index], y_labels[test_index]
    clf.fit(X_train, y_train)
    loo_scores.append(accuracy_score(y_test, clf.predict(X_test)))

loo_mean_accuracy = mean(loo_scores)
loo_std_accuracy = stdev(loo_scores)
print("Precisión media (leave-one-out):", loo_mean_accuracy)
print("Desviación estándar (leave-one-out):", loo_std_accuracy)

# Validación cruzada leave-k-out con k=2
k = 2
k_scores = []
k_cv = KFold(n_splits=len(X) - k)

for train_index, test_index in k_cv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_labels[train_index], y_labels[test_index]
    clf.fit(X_train, y_train)
    k_scores.append(accuracy_score(y_test, clf.predict(X_test)))

k_mean_accuracy = mean(k_scores)
k_std_accuracy = stdev(k_scores)
print(f"Precisión media (leave-{k}-out):", k_mean_accuracy)
print(f"Desviación estándar (leave-{k}-out):", k_std_accuracy)
