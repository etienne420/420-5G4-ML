import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('future.no_silent_downcasting', True)

nosPokemons = pd.read_csv("apprentissage/datas/pokedex.csv")

#Affichage des colonnes du Dataframe
print(nosPokemons.columns.values)

print(nosPokemons.head(10))

# remplacement de VRAI FAUX par 1 0
nosPokemonsClean = nosPokemons
nosPokemonsClean["LEGENDAIRE"] = nosPokemonsClean["LEGENDAIRE"].replace("VRAI", 1)
nosPokemonsClean["LEGENDAIRE"] = nosPokemonsClean["LEGENDAIRE"].replace("FAUX", 0)
nosPokemonsClean["LEGENDAIRE"] = nosPokemonsClean["LEGENDAIRE"].astype(int)
print(nosPokemonsClean.head())

#Comptage du nombre d'observations et de features
print(nosPokemonsClean.shape)

#Informations sur notre jeu de données
print(nosPokemonsClean.info())

print(nosPokemonsClean[nosPokemonsClean['NOM'].isnull()])

print(nosPokemonsClean['NOM'][61])
print(nosPokemonsClean['NOM'][63])

nosPokemonsClean.loc[62, 'NOM'] = "Colossinge"

print(nosPokemonsClean['NOM'][62])

#Chargement des données des combats
combats = pd.read_csv("apprentissage/datas/combats.csv")

#Affichage des colonnes du Dataframe
print(combats.columns.values)

#Affichage des 10 premières lignes du Dataframe
print(combats.head(10))

#Comptage du nombre de lignes et de colonnes
print (combats.shape)

#Informations sur notre jeu de données
print (combats.info())

nbFoisPremierePosition = combats.groupby('Premier_Pokemon').count()
print(nbFoisPremierePosition)

nbFoisSecondePosition = combats.groupby('Second_Pokemon').count()
print(nbFoisSecondePosition)

nombreTotalDeCombats = nbFoisPremierePosition + nbFoisSecondePosition
print(nombreTotalDeCombats)

nombreDeVictoires = combats.groupby('Pokemon_Gagnant').count()
print(nombreDeVictoires)

listeAAgreger = combats.groupby('Pokemon_Gagnant').count()
listeAAgreger.sort_index()

#On ajoute le nombre de combats
listeAAgreger['NBR_COMBATS'] = nbFoisPremierePosition.Pokemon_Gagnant + nbFoisSecondePosition.Pokemon_Gagnant

#On ajoute le nombre de victoires
listeAAgreger['NBR_VICTOIRES'] = nombreDeVictoires.Premier_Pokemon

#On calcule le pourcentage de victoires
listeAAgreger['POURCENTAGE_DE_VICTOIRES'] = nombreDeVictoires.Premier_Pokemon / (nbFoisPremierePosition.Pokemon_Gagnant + nbFoisSecondePosition.Pokemon_Gagnant)

#On affiche la nouvelle liste
print(listeAAgreger)

nouveauPokedex = nosPokemonsClean.merge(listeAAgreger, left_on='NUMERO', right_index = True, how='left')
print(nouveauPokedex.describe())

axe_X = sns.countplot(x="TYPE_1", hue="LEGENDAIRE", data=nouveauPokedex)
plt.xticks(rotation= 90)
plt.xlabel('TYPE_1')
plt.ylabel('Total ')
plt.title("POKEMONS DE TYPE_1")
plt.show()

axe_X = sns.countplot(x="TYPE_2", hue="LEGENDAIRE", data=nouveauPokedex)
plt.xticks(rotation= 90)
plt.xlabel('TYPE_2')
plt.ylabel('Total ')
plt.title("POKEMONS DE TYPE_2")
plt.show()

print(nouveauPokedex.groupby('TYPE_1').agg({"POURCENTAGE_DE_VICTOIRES": "mean"}).sort_values(by = "POURCENTAGE_DE_VICTOIRES"))

corr = nouveauPokedex.loc[:,['POINTS_DE_VIE','NIVEAU_ATTAQUE',
'NIVEAU_DEFENSE','NIVEAU_ATTAQUE_SPECIALE','NIVEAU_DEFENSE_SPECIALE',
'VITESSE','LEGENDAIRE','POURCENTAGE_DE_VICTOIRES']].corr()

sns.heatmap(corr,
xticklabels=corr.columns,
yticklabels=corr.columns, annot=True)
plt.show()

#Sauvegarde du Dataframe Pokedex
dataset = nouveauPokedex
dataset.to_csv("datas/dataset.csv", sep='\t')

#Chargement du dataset avec pour séparateur des tabulations
dataset = pnd.read_csv("datas/dataset.csv",delimiter='\t')

#Suppression des lignes ayant des valeurs manquantes
dataset = dataset.dropna(axis=0, how='any')

#X = on prend toutes les données, mais uniquement les features 5 à 12 
#(ne pas hésiter à ouvrir le fichier afin de bien visualiser les features concernées)
#POINTS_ATTAQUE;POINTS_DEFFENCE;POINTS_ATTAQUE_SPECIALE;POINT_DEFENSE_SPECIALE;POINTS_VITESSE;NOMBRE_GENERATIONS
X = dataset.iloc[:, 5:12].values

#y = on prend uniquement la feature POURCENTAGE_DE_VICTOIRE (17ème feature)
y = dataset.iloc[:, 17].values

#X = on prend toutes les données, mais uniquement les features 4 à 11
# POINTS_ATTAQUE;POINTS_DEFFENCE;POINTS_ATTAQUE_SPECIALE;POINT_DEFENSE_SPECIALE;POINTS_VITESSE;NOMBRE_GENERATIONS
X = dataset.iloc[:, 5:12].values

#y = on prend uniquement la colonne POURCENTAGE_DE_VICTOIRE (16ème feature) 
#les : signifiant "Pour toutes les observations"
y = dataset.iloc[:, 16].values

#Construction du jeu d'entrainement et du jeu de tests
from sklearn.model_selection import train_test_split
X_APPRENTISSAGE, X_VALIDATION, Y_APPRENTISSAGE, Y_VALIDATION = train_test_split(X, y, test_size = 0.2, random_state = 0)

#---- ALGORITHME 1: REGRESSION LINEAIRE -----
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

#Choix de l'algorithme
algorithme = LinearRegression()

#Apprentissage à l'aide de la fonction fit
#Apprentisage effectué sur le jeu de données d'apprentissage
algorithme.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)

#Realisation de prédictions sur le jeu de tests (validation)
predictions = algorithme.predict(X_VALIDATION)

#Calcul de la précision de l'apprentissage à l'aide de la
#fonction r2_score en comparant les valeurs prédites
#(predictions) et les valeurs attendues (Y_VALIDATION)
precision = r2_score(Y_VALIDATION, predictions)

print(">> ----------- REGRESSION LINEAIRE -----------")
print(">> Precision = "+str(precision))
print("------------------------------------------")

#Choix de l'algorithme
from sklearn.tree import DecisionTreeRegressor
algorithme = DecisionTreeRegressor()
algorithme.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)
predictions = algorithme.predict(X_VALIDATION)
precision = r2_score(Y_VALIDATION, predictions)

print(">> ----------- ARBRES DE DECISIONS -----------")
print(">> Precision = "+str(precision))
print("------------------------------------------")

#Choix de l'algorithme
from sklearn.ensemble import RandomForestRegressor
algorithme = RandomForestRegressor()
algorithme.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)
predictions = algorithme.predict(X_VALIDATION)
precision = r2_score(Y_VALIDATION, predictions)

print(">> ----------- FORETS ALEATOIRES -----------")
print(">> Precision = "+str(precision))
print("------------------------------------------")

import joblib
fichier = 'modele/modele_pokemon.mod'
joblib.dump(algorithme, fichier)

predictions = algorithme.predict(X_VALIDATION)
precision_apprentissage = algorithme.score(X_APPRENTISSAGE,Y_APPRENTISSAGE)
precision = r2_score(Y_VALIDATION, predictions)

#Module de lecture de fichiers CSV
import csv
#Module de chargement du modèle d'apprentissage
import joblib

def rechercheInformationsPokemon(numPokemon,Pokedex):
    infosPokemon = []
    for pokemon in Pokedex:
        if (int(pokemon[0])==numPokemon):
            infosPokemon = [pokemon[0],pokemon[1],pokemon[4],pokemon[5],pokemon[6],pokemon[7],pokemon[8],pokemon[9],pokemon[10]]
            break
    return infosPokemon

def prediction (numeroPokemon1, numeroPokemon2,Pokedex):
    pokemon1 = rechercheInformationsPokemon(numeroPokemon1, Pokedex)
    pokemon2 = rechercheInformationsPokemon(numeroPokemon2, Pokedex)
    
    modele_prediction = joblib.load('modele/modele_pokemon.mod')
    
    prediction_Pokemon_1 = modele_prediction.predict([[pokemon1[2],pokemon1[3],pokemon1[4],pokemon1[5],pokemon1[6],pokemon1[7],pokemon1[8]]])
    
    prediction_Pokemon_2 = modele_prediction.predict([[pokemon2[2], pokemon2[3], pokemon2[4], pokemon2[5], pokemon2[6], pokemon2[7], pokemon2[8]]])
    
    print ("COMBAT OPPOSANT : ("+str(numeroPokemon1)+") "+pokemon1[1]+" à ("+str(numeroPokemon2)+") "+pokemon2[1])
    print (" "+pokemon1[1]+": "+str(prediction_Pokemon_1[0]))
    print(" " +pokemon2[1] + ": " + str(prediction_Pokemon_2[0]))
    print ("")
    
    if prediction_Pokemon_1>prediction_Pokemon_2:
        print(pokemon1[1].upper()+" EST LE VAINQUEUR !")
    else:
        print(pokemon2[1].upper() + " EST LE VAINQUEUR !")

        modele_prediction = joblib.load('modele/modele_pokemon.mod')

        #Chargement du Pokédex et lancement d'un combat
with open("datas/pokedex.csv", newline='') as csvfile:
    pokedex = csv.reader(csvfile)
    next(pokedex)
    prediction(368,598,pokedex)