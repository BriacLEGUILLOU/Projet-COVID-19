# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
pip install seaborn --user
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_row', 111)
pd.set_option('display.max_column', 111)

url = 'https://raw.githubusercontent.com/MachineLearnia/Python-Machine-Learning/master/Dataset/dataset.csv'
data = pd.read_csv(url, index_col=0, encoding = "ISO-8859-1")

data.head()

############################################
# 1. Exploratory Data Analysis
############################################
"""
Objectif :

    Comprendre du mieux possible nos données (un petit pas en avant vaut mieux qu'un grand pas en arriere)
    Développer une premiere stratégie de modélisation

Checklist de base
Analyse de Forme :
    
    variable target : SARS-Cov-2 exam result
    lignes et colonnes : 5644, 111
    types de variables : qualitatives : 70, quantitatives : 41
    Analyse des valeurs manquantes :
        beaucoup de NaN (la moitié des variables ont 90% de NaN)
        groupe de donnée à 76% de valeur manquantes -> test viral
        groupe de donnée à 89% ve valeur manquantes -> taux sanguins

Analyse de Fond :

    Visualisation de la target :
        90% de cas négatif
        10% de cas positif

    Signification des variables :
        variables continues standardisées, skewed (asymétriques), test sanguin
        age quantile : difficile d'interpreter ce graphique, clairement ces données ont été traitées, 
        on pourrait penser 0-5, mais cela pourrait aussi etre une transformation mathématique. 
        On peut pas savoir car la personne qui a mit ce dataset ne le précise nul part. 
        variable qualitative : binaire (0, 1), viral, Rhinovirus qui semble tres élevée



    Relation Variables / Target :
        * target / blood : les taux de Monocytes, Platelets, Leukocytes semblent liés au covid-19 -> hypothese a tester
        * target/age : les individus de faible age sont tres peu contaminés ? 
        -> attention on ne connait pas l'age, et on ne sait pas de quand date le dataset 
        (s'il s'agit des enfants on sait que les enfants sont touchés autant que les adultes). 
        En revanche cette variable pourra etre intéressante pour la comparer avec les résultats de tests sanguins
        * target / viral : les doubles maladies sont tres rares. Rhinovirus/Enterovirus positif 
        - covid-19 négatif ? -> hypothese a tester ? mais il est possible que la région 
        est subie une épidémie de ce virus. De plus on peut tres bien avoir 2 virus en meme temps. 
        Tout ca n'a aucun lien avec le covid-19


Conclusions intiales :
    * Beaucoup de données manquantes (on peut garder au mieux 20% du dataset)
    * 2 groupes de données intéressates (viral, sanguin)
    Peu de variable "discriminante" pour distinguer les cas positifs/négatifs.
    Il n'est donc pas possible de prédire qu'un individu est atteint du covid en se basant sur des tests sanguins.
    Nous devons soit prédire les malades, soit montrer qu'il n'est pas possible de le prédire.



Analyse plus détaillée
* Relation Variables / Variables
  - blood_data / blood_data : certaines variables sont très corrélées : +0.9 (à surveiller plus tard)
  - blood_data / age : très faible corrélation entre l'age et taux sanguins
  - viral / viral : Influenza rapid test donne de mauvais résultats, il faudra peut-être la laisser tomber
  - relation maladie / blood_data
  - relation hospitalisation / est malade
  - relation hospitalisation / blood : intéressant dans le cas ou il faut prédire le service de la personne

* NaN analyse

Hypothèse nulle (H0):
    * Les individus atteint du covid-19 ont des taux de
    Leukocytes, Monocytes, Platelets significativement différents
    Hypothèse rejeté : Les taux moyens sont égaux chez les individus positifs et négatifs
    * Les individus atteint d'une quelconque maladie ont des taux significativement différent


"""

##################################
# Analyse de la forme des données
df = data.copy()

df.shape

df.dtypes.value_counts()

# Visualiser les valeurs manquantes
plt.figure(figsize=(15,10))
sns.heatmap(df.isna())
""" Obersation: on observe beaucoup de valeurs manquantes
Il semble que les mv soient réparties par ligne"""

(df.isna().sum() / df.shape[0]).sort_values(ascending=True)


##################################
# Analyse du Fond
# 1. Visualisatrion initial- Elimination des colonnes inutiles
df = df[df.columns[df.isna().sum() / df.shape[0] < 0.9]]
#df = df.drop('Patient ID', axis=1)


# Examen de la colonne target
df['SARS-Cov-2 exam result'].value_counts(normalize=True)

# Histogrames des variables continues
for col in df.select_dtypes(float):
    plt.figure()
    sns.distplot(df[col]) #Distribution plot

""" Obersations : il semble que les données ont été standardisées
La plupart des variables suivent des distributions normales
Certaines distributions ont des distributions asymétiques
"""

sns.distplot(df['Patient age quantile'])
""" Il semble que les données de l'âge ont été modifié"""

# Variable Qualitatives
df['SARS-Cov-2 exam result']

for col in df.select_dtypes(object):
    #f string
    print(f'{col :-<50} {df[col].unique()}')

""" Ce sont des variables binaires """

for col in df.select_dtypes(object):
    plt.figure()
    df[col].value_counts().plot.pie()

""" La majorité des test sont négatifs, sauf Rhinovirus """


# Relation variable / Target
## Création de sous-ensembles positifs et négatifs
positive_df = df[df['SARS-Cov-2 exam result'] == 'positive']
negative_df = df[df['SARS-Cov-2 exam result'] == 'negative']

## Création de sous-ensembles Blood et viral
missing_rate = df.isna().sum()/df.shape[0]
blood_columns = df.columns[(missing_rate < 0.9) & (missing_rate >0.88)]
viral_columns = df.columns[(missing_rate < 0.88) & (missing_rate > 0.75)]

## Target / Blood
for col in blood_columns:
    sns.distplot(positive_df[col], label='positive')
    sns.distplot(negative_df[col], label='negative')
    plt.legend()
    plt.plot()
    plt.show()

""" On peut maitenant voir pour chaque variable s'il existe une différence 
entre les cas positifs et négatifs 
- Platelets : il semble que les gens positifs ont des taux de platelets différents
C'est une idée, il faudra tester cette hypothèse
-Leukcytes
-Monocytes
"""

## Relation Target / age
sns.countplot(x='Patient age quantile', hue='SARS-Cov-2 exam result', data=df)

""" Il semble que l'âge soit lié au fait d'être positif ou négatif
Malheuresement, on ne connait pas trop la variable âge """


# Relation Target / Viral
pd.crosstab(df['SARS-Cov-2 exam result'], df['Influenza A'])

for col in viral_columns:
    plt.figure()
    sns.heatmap(pd.crosstab(df['SARS-Cov-2 exam result'], df[col]), annot=True, fmt='d')
    plt.plot()
    plt.show()

""" On peut observer qu'il est arre d'avoir 2 virus en même temps 
Beaucoup de gens avec le Rhinovirus n'ont pas le covid"""

#################################################
# Analyse plus détaillée
# Relation variable / variable par catégorie

#################################################
#Analyse plus détaillée
## Relation Variables / Variables

# Relation Taux Sanguin
sns.pairplot(df[blood_columns])
sns.heatmap(df[blood_columns].corr())
sns.clustermap(df[blood_columns].corr())

# Relation blood_data / age
for col in blood_columns:
    plt.figure()
    # Permet de visualiser des courbes de regression dans les nuages de points
    sns.lmplot(x='Patient age quantile', y=col, hue='SARS-Cov-2 exam result', data=df)
    plt.show()
    
""" On cherche à observer une relation linéaire entre l'age et certains taux sanguins
Il semble que la réponse soit non
"""

# On affiche la corrélation entre l'age et les colonnes sang
df.corr()['Patient age quantile'].sort_values()
""" les corrélations les plus élevés atteignent 0.28, ce qui est très faible"""

# relation entre les variables virales
pd.crosstab(df['Influenza A'], df['Influenza A, rapid test'])

pd.crosstab(df['Influenza B'], df['Influenza B, rapid test'])

""" On fait des recherches sur Internet pour trouver que ces tests ne sont pas fiables """


# Relation être malade et les taux de globule blanc / rouge
# Création d'une nouvelle variable est "malade"
df['est malade'] = np.sum(df[viral_columns[:-2]] == 'detected', axis=1) >= 1

# Pour viasualiser la relation entre maladie et les taux, on créer des sous-ensembles
malade_df = df[df['est malade'] == True]
non_malade_df = df[df['est malade'] == False]

for col in blood_columns:
    sns.distplot(malade_df[col], label='malade')
    sns.distplot(non_malade_df[col], label='non malade')
    plt.legend()
    plt.plot()
    plt.show()

""" On va essayer de voir si les taux qui étaient discriminants pour le covid sont aussi différents
Les taux de Monocytes, Platelets, Leukocytes semblent liés au covid-19 
les taux de lymphocytes semblent différents pour 'est malade'
 """

def hospitalisation(df):
    if df['Patient addmited to regular ward (1=yes, 0=no)'] == 1:
        return 'surveillance'
    elif df['Patient addmited to semi-intensive unit (1=yes, 0=no)'] == 1:
        return 'soins semi-intensives'
    elif df['Patient addmited to intensive care unit (1=yes, 0=no)'] == 1:
        return 'soins intensifs'
    else:
        return 'inconnu'

df['statut'] = df.apply(hospitalisation, axis=1)
df['statut'].value_counts()

for col in blood_columns:
    plt.figure()
    for cat in df['statut'].unique():
        sns.distplot(df[df['statut']==cat][col], label=cat)
    plt.legend()



df.dropna().count() # dans le cas ou on élimine les vm, il ne nous reste que 99 lignes
df[viral_columns].count() # 1354 lignes si on travaille avec viral columns
df[blood_columns].count() #602 columns si on travail avec blood columns

""" il semble qu'il vaut mieux travailler avec viral columns
Mais cela ne suffira pas, les blood columns sont extrêment utiles.

Pour terminer l'analyse avec les valeurs manquantes", il faut essayer de comprendre :
    'Quelle est l'état de la target si on élimine toutes les colonnes de blood_columns ou viral_columns ?""
"""
df2 = df[viral_columns]
df2['covid'] = df['SARS-Cov-2 exam result']
df2.dropna(inplace=True)
df2['covid'].value_counts()

df2 = df[blood_columns]
df2['covid'] = df['SARS-Cov-2 exam result']
df2.dropna(inplace=True)
df2['covid'].value_counts()


#####################################################
# Hypothèse nulle
"""
Hypothèse nulle (H0):
    * Les individus atteint du covid-19 ont des taux de
    Leukocytes, Monocytes, Platelets significativement différents
    * Les individus atteint d'une quelconque maladie ont des taux significativement différent
    
Le test de Student permet de tester si la moyenne entre 2 échantillons est significativement différente
Le principe de Student va être d'essayer de rejeter l'hypothèse d'égalité des moyennes
On définit un seuil alpha, si la valeur p est inférieur au seuil alpha:
    alors on peut rejeter cette hypothèse, sinon on ne peut pas la rejeter

Pour faire un T-Test, il est préférable d'avoir des classes équilibrés."""


# T-Test
from scipy.stats import ttest_ind
positive_df.shape
negative_df.shape
negative_df.sample(len(positive_df))
balanced_neg = negative_df.sample(positive_df.shape[0])

def t_test(col):
    alpha = 0.02
    stat, p = ttest_ind(balanced_neg[col].dropna(), positive_df[col].dropna())
    if p < alpha:
        return 'H0 Rejetée'
    else :
        return 0


for col in blood_columns:
    print(f'{col :-<50} {t_test(col)}')


    """ Observations Les Platelets, les Leucocytes et les Eosinophils semblent être des variables significativement différentes
    pour les gens atteint du covid"""

