# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 10:50:27 2022

@author: antony
"""


import statistics
import pandas as pd
import numpy as np
import math 
import matplotlib.pyplot as plt 
from tqdm import tqdm
"""
 Pour chaque vecteur , on effectue une épurarion des temps trop absurde 
 pour être intégré dans le modèle. Cela permet de lisser les données et d'obtenir
 une moyenne plus cohérente pour le modèle futur. Enfin , on élimine les vecteurs
 au temps trop différent de la moyenne. 
"""
pd.options.mode.chained_assignment = None  # default='warn'
sheet=pd.read_pickle("data_light_reduced.pkl")
liste_temps=pd.read_pickle("liste_temps_light.pkl")
def smoothing_means(df,column):
    """
    Elimine les temps absurdes utilisés pour chaque vecteur 
    """
    print("Lissage en cours ...\n")
    l = column
    #print("l",l)
    #print("dropping",liste_temps.iloc[:,l].dropna(),[75,25])
    q75,q25 = np.percentile(df.iloc[:,l].dropna(),[75,25])
    intr_qr = q75-q25
    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)
    filter = (df.iloc[:,l].dropna() >= min) & (liste_temps.iloc[:,l].dropna()<=max) #ce filtre garde l'information sur les index 
    #print(filter)
    #print()
    #print(sheet['index'].iloc[l])
    #print()
    df=df[filter]
    sheet['index'].iloc[l]=pd.DataFrame(sheet['index'].iloc[l])[filter].T.values.tolist()[0] #on conserve uniquement les index encore présents
    #print("les temps sont", liste_temps.iloc[:,l].dropna()[filter])
    #print("filtré", sheet_1['time'].loc[sheet['index'].iloc[l]])
    #print()
    liste_temps.iloc[:,l]=liste_temps.iloc[:,l].dropna().loc[filter].dropna() #on conserve les bon index ! ainsi on peut savoir qui a été épuré
    
    #print("now",liste_temps.iloc[:,l].dropna())
    info=[]
    for i in range(0,len(liste_temps.columns)) : #add data that need a len >2
         L=liste_temps[i].dropna()
         if len(L) >2:
             quantiles  = np.quantile(L,[0.25,0.5,0.75])
             info.append([math.sqrt(statistics.variance(L)),quantiles[0],quantiles[2]])
         else : 
             info.append(3*[0])
    #actualisation du sheet   
    sheet["time"]=pd.DataFrame([statistics.mean(liste_temps[i].dropna()) for i in range(0,len(liste_temps.columns))])
    sheet['min']=pd.DataFrame([liste_temps[i].dropna().min() for i in range(0,len(liste_temps.columns))])
    sheet['max']=pd.DataFrame([liste_temps[i].dropna().max() for i in range(0,len(liste_temps.columns))])
    sheet['median']=pd.DataFrame([statistics.median(liste_temps[i].dropna()) for i in range(0,len(liste_temps.columns))])
    sheet['écart-type']=pd.DataFrame([info[i][0] for i in range(0,len(info))])
    sheet['25%']=pd.DataFrame([info[i][1] for i in range(0,len(info))])
    sheet['75%']=pd.DataFrame([info[i][2] for i in range(0,len(info))])
    sheet['lenght']=pd.DataFrame([len(liste_temps[i].dropna()) for i in range(0,len(liste_temps.columns))])
    return sheet
    
    #on se sépare des vecteurs basés sur un pannel de temps trop conséquent , on crée également les classes de temps relativement 
    #à ces données. 
def prunning(sheet,sheet_2):
    """
    Elimine les vecteurs au temps trop éloigné de la moyenne
    """
    print("Epuration en cours ...\n")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle('Profil des écart inter-quartiles des temps moyenné de nos vecteurs pour chaque set')
    red_circle = dict(markerfacecolor='red', marker='o', markeredgecolor='white')
    ax1.boxplot(sheet_2['75%']-sheet_2['25%'], flierprops=red_circle) 
    ax1.set_title("Statut initial")
    ax2.boxplot(sheet['75%']-sheet['25%'], flierprops=red_circle) 
    ax2.set_title("Après lissage")
    ax2.set( xlabel='75%-25%')

    ax1.set(ylabel='tps(ms)', xlabel='75%-25%')
    q75,q25 = np.percentile(sheet['75%']-sheet['25%'],[75,25])
    intr_qr = q75-q25
    max = q75+(1.5*intr_qr)
    filter = sheet['75%']-sheet['25%']<=max
    sheet=sheet[filter]
    ##nouveau set retrié
    IQR=np.quantile(sheet['75%']-sheet['25%'],0.75)-np.quantile(sheet['75%']-sheet['25%'],0.25)
    Q3= np.quantile(sheet['75%']-sheet['25%'],0.75)#prendre le maximum , les classes pourraient être trop nombreuses
    security = int(Q3 + (1.5 * IQR))
    ax3.boxplot(sheet['75%']-sheet['25%'], flierprops=red_circle)
    ax3.plot(1,security,marker="o",markerfacecolor="green",label="security")
    ax3.set_title("Après épuration")
    ax3.set( xlabel='75%-25%')
    ax3.legend()
    plt.tight_layout()
    return sheet
def smooth (sheet,liste_temps):
    """
    Main function
    """
    sheet_2=sheet.copy()
    print("Nous comptions %s éléments \n" %len(sheet))
    sheet = smoothing_means(sheet,liste_temps)   
    sheet = prunning(sheet,sheet_2)
    print("Nous en comptons maintenant %s soit une réduction de %d" %(len(sheet),100-len(sheet)/len(sheet_2)*100) +'%' )
    return sheet
                                           
sheet=smooth(sheet,liste_temps)