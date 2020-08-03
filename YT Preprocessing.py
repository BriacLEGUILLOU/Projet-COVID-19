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

print()
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



def imputation(df):
    #df['is na'] = (df['Parainfluenza 3'].isna()) | (df['Leukocytes'].isna())
    #df = df.fillna(-999)
    df = df.dropna(axis=0)
    return  df





############################
# Modèle basic
def preprocessing_basic(df):
    """ 1ère version du pré-traitement de données """
    df = encodage(df)
    df = imputation(df)
    
    X =df.drop('SARS-Cov-2 exam result', axis=1)
    y = df['SARS-Cov-2 exam result']
    print(y.value_counts())
    return X, y

X_train, y_train = preprocessing_basic(trainset)
X_test, y_test = preprocessing_basic(testset)
""" Ceci est la version la plus basique du pré-traitement de données """

# Modélisation
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=0)

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
    

evaluation(model)
""" Sur la courbe d'apprentissage
train_score = 100%
val_score < 50%
Donc on est en overfitting 

Pour lutter contre l'overfitting :
    - on peut fournir plus de données dropna() ->
    Le preprocessing consiste à explorer différentes options pour améliorer notre modèle"""








