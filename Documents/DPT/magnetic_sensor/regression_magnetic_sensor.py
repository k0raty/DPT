# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 13:42:16 2022

@author: anton
"""
import pandas as pd

df=pd.read_excel('Sensor.xlsx')
#selection de la première colonne de notre dataset (la taille de la population)
X = df.iloc[0:len(df),0]
#selection de deuxième colonnes de notre dataset (le profit effectué)
Y = df.iloc[0:len(df),1] 
 
#Affichage#
import matplotlib.pyplot as plt
 
axes = plt.axes()
axes.grid() # dessiner une grille pour une meilleur lisibilité du graphe
plt.scatter(X,Y) # X et Y sont les variables qu'on a extraite dans le paragraphe précédent
plt.show()

#Regression#
from scipy import stats
import numpy as np
mymodel = np.poly1d(np.polyfit(X, Y, 2))
#linregress() renvoie plusieurs variables de retour. On s'interessera 
# particulierement au slope et intercept
myline = np.linspace(0, 3, 100)

plt.scatter(X, Y)
plt.plot(myline, mymodel(myline))
plt.show() 
