#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 08:08:25 2020

@author: leguilloubriac
pip install sklearn --user
SARS-Cov-2 exam result
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



url = 'https://raw.githubusercontent.com/MachineLearnia/Python-Machine-Learning/master/Dataset/dataset.csv'
data = pd.read_csv(url, encoding = "ISO-8859-1")
data.head()

######################################
# PRE-PROCESSING
df = data.copy()
df.head()

# Création des sous-ensembles (suite au EDA)
missing_rate = df.isna().sum()/df.shape[0]

blood_columns = list(df.columns[(missing_rate < 0.9) & (missing_rate >0.88)])
viral_columns = list(df.columns[(missing_rate < 0.80) & (missing_rate > 0.75)])

key_columns = ['Patient age quantile', 'SARS-Cov-2 exam result']

df = df[key_columns + blood_columns + viral_columns]
df.head()


# TrainTest - Nettoyage - Encodage
from sklearn.model_selection import train_test_split
trainset, testset = train_test_split(df, test_size=0.2, random_state=0)

trainset['SARS-Cov-2 exam result'].value_counts()

testset['SARS-Cov-2 exam result'].value_counts()


def encodage(df):
    """ Permet de numériser les colonnes """
    code = {'negative':0,
            'positive':1,
            'not_detected':0,
            'detected':1}
    
    for col in df.select_dtypes('object').columns:
        df.loc[:,col] = df[col].map(code)
        
    return df


def feature_engineering(df):
    df['est malade'] = df[viral_columns].sum(axis=1) >= 1
    df = df.drop(viral_columns, axis=1)
    return df

def imputation(df):
    #df['is na'] = (df['Parainfluenza 3'].isna()) | (df['Leukocytes'].isna())
    #df = df.fillna(-999)
    df = df.dropna(axis=0)
    return  df

""" fillna couplé à un indicateur de valeur manquantes ne fonctionne pas """




############################
# Modèle basic
def preprocessing(df):
    """ 1ère version du pré-traitement de données """
    df = encodage(df)
    df = feature_engineering(df)
    df = imputation(df)
    
    X =df.drop('SARS-Cov-2 exam result', axis=1)
    y = df['SARS-Cov-2 exam result']
    print(y.value_counts())
    return X, y

X_train, y_train = preprocessing(trainset)
X_test, y_test = preprocessing(testset)
""" Ceci est la version la plus basique du pré-traitement de données """

# Procédure d'évaluation
from sklearn.metrics import f1_score, confusion_matrix, classification_report
# bonne métric pour avoir le rapport entre la précision et le recall
# permet de mesurer les proportions de faux postifs et de faux négatifs
from sklearn.model_selection import learning_curve # permet de détecter overfitting et underfitting

def evaluation(model):
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)
    
    print(confusion_matrix(y_test, ypred))
    print(classification_report(y_test, ypred))
    
    # AJout de learning surve, pour identifier overfitting et underfitting
    N, train_score, val_score = learning_curve(model, X_train, y_train,
                                               cv=4, scoring='f1',
                                               train_sizes=np.linspace(0.1, 1, 10)) 
    # learning_curve utilise validation croisée pour entrainer et évaluer le modèle sur plusieurs split du dataset
    
    # Visualisation de la learning curve
    plt.figure(figsize=(12,8))
    plt.plot(N, train_score.mean(axis=1), label='train_score')
    plt.plot(N, val_score.mean(axis=1), label='test_score')
    plt.xlabel('trainsize')
    plt.ylabel('f1 score')
    plt.legend()
    
# Modélisation
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import PolynomialFeatures, StandardScaler




preprocessor = make_pipeline(PolynomialFeatures(2, include_bias=False), SelectKBest(f_classif, k=10))

RandomForest = make_pipeline(preprocessor, RandomForestClassifier(random_state=0))
AdaBoost = make_pipeline(preprocessor, AdaBoostClassifier(random_state=0))
SVM = make_pipeline(preprocessor, StandardScaler(), SVC(random_state=0))
KNN = make_pipeline(preprocessor, StandardScaler(), KNeighborsClassifier())

dict_of_models = {'RandomForest' : RandomForest, 
                  'AdaBoost' : AdaBoost, 
                  'SVM' : SVM, 
                  'KNN' : KNN}

for name, model in dict_of_models.items():
    print(name)
    evaluation(model)




""" 
Ce qui est important, c'est d'avoir un écart réduit dans les courbes d'apprentissages
entre le train et le validation.
Cela montre que le modèle a biren appris et qu'il est capable de généraliser.
Ici, SVM et KNN semblent prometteurs.
 """
 
 
############################################################
 # Optimisation
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
""" Pour utiliser GridSearchCV, il faut définir un dictionnaire d'Hyperparamètres.
 Pour connaitre les différents hyperparamètres, on affiche toute la pipeline en entier. 
 Check the list of available parameters with `estimator.get_params().keys()`"""
print(SVM.get_params())

hyper_params = {'svc__gamma':[1e-3, 1e-4], 
                'svc__C':[1, 10, 100, 1000]}

grid = GridSearchCV(SVM, hyper_params, scoring='recall', cv=4)

grid.fit(X_train, y_train)

print(grid.best_params_)

y_pred = grid.predict(X_test)

print(classification_report(y_test, y_pred))

evaluation(grid.best_estimator_)


""" On pourrait être tenté de faire une GridSearCV sur tous les paramètres.
On peut utiliser RandomSearchCV """


hyper_params = {'svc__gamma':[1e-3, 1e-4], 
                'svc__C':[1, 10, 100, 1000],
                'pipeline__polynomialfeatures__degree':[2, 3, 4],
                 'pipeline__selectkbest__k': range(4, 100)}

grid = RandomizedSearchCV(SVM, hyper_params, scoring='recall', cv=4,
                          n_iter=40)

grid.fit(X_train, y_train)

print(grid.best_params_)

y_pred = grid.predict(X_test)

print(classification_report(y_test, y_pred))

evaluation(grid.best_estimator_)

""" On réduit ensuite le range de nos paramètres """
hyper_params = {'svc__gamma':[1e-3, 1e-4], 
                'svc__C':[1, 10, 100, 1000],
                'pipeline__polynomialfeatures__degree':[2, 3, 4],
                 'pipeline__selectkbest__k': range(40, 80)}




##############################################
# Precision Recall Curve
""" Pour finaliser la création de notre modèle, on va observer les courbes précisions
et recall. on va définir un seuil de prédiction pour notre modèle.
La fonction precision_recall_curve permet de visualiser précision ou sensibilité
de notre modèle en fonction d'un seuil de précision définit.

La plupart des modèle ont une fonction de décision. (régression logistique, 
p(X)<0.5 -> 0
p(X) >= 0.5 -> 1
L'impact de ce seuil, on peut le visualiser avec precision_recall_curve"""
from sklearn.metrics import precision_recall_curve

precision, recall, threshold = precision_recall_curve(y_test, grid.best_estimator_.decision_function(X_test))

plt.plot(threshold, precision[:-1], label='precision')
plt.plot(threshold, recall[:-1], label='recall')
plt.legend()
plt.show()

""" Observation:
    Si on prends un thrshod de -12, alors le recall se rapproche de 100
    on a donc identifié toutes les personnes atteintes du Coronavirus
    Mais la précision est mauvaise.
    Il est important de trouver un bon équilibre entre la précision et le recall.

Plus la précision augmente, plus le recall diminue.
L'idée est de trouver le seuil de décision avec le meilleur recall/précision: score f1

Selon notre cahier des charges, on peut sacrifier un peu de recall pour avoir de la précision
"""


def model_final(model, X, threshold=0):
    return model.decision_function(X) > threshold

y_pred = model_final(grid.best_estimator_, X_test, threshold=1)

f1_score(y_test, y_pred)

from sklearn.metrics import recall_score

recall_score(y_test, y_pred)

""" Lors de la modélisation, il faut tester plusierus modèles.
Evaluer tous ces modèles avec une procédure robuste

Retenir les modèles avec les meilleurs performances et optimiser ce modèle.

Dans le cadre d'une classification binaire, on peut afficher 
la courbe precision/recall.
"""
evaluation(grid.best_estimator_)



