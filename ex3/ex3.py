import pandas as pd
import numpy as np



def balance_dataset(data):
    #Contagem das classes
    classes_col = data[:, 0]
    classes, classes_counts = np.unique(classes_col, return_counts=True)
    biggest_class_count = np.max(classes_counts)

    #Pegar os dados por classe
    classes_vects = [data[classes_col == cur_class, :] for cur_class in classes]
    #Selecionar aleatoriamente quais samples serao repetidas
    choices = [np.random.choice(len(class_vect), biggest_class_count-len(class_vect)) for class_vect in classes_vects]
    #Concatenar as samples repetidas com as originais para balancear o dataset
    classes_vects = [np.concatenate([class_vect, class_vect[choice]]) for class_vect, choice in zip(classes_vects, choices)]
    #Reconstroi o dataset
    new_data = np.concatenate(classes_vects)
    #Embaralha o dataset (pois o processo do balanceamento ordena por classe)
    np.random.shuffle(new_data)

    return new_data



#Leitura dos dados
cols = ["Class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", 
        "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", 
        "Color intensity", "Hue","OD280/OD315 of diluted wines", "Proline"]
df = pd.read_csv("wine.data", header=None, index_col=False, names=cols)

test_split = 0.20
data = df.values
#Normalizacao
data[:, 1:] = (data[:, 1:] - np.mean(data[:, 1:], axis=0)) / np.std(data[:, 1:], axis=0)
#Embaralhar dataset
np.random.shuffle(data)
#Separar treino testo
test = data[:int(len(data)*test_split), :]
train = data[int(len(data)*test_split):, :]
#Balancear os datasets
test = balance_dataset(test)
train = balance_dataset(train)
print(np.unique(train[:, 0], return_counts=True)[1], np.unique(test[:, 0], return_counts=True)[1])