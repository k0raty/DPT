# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 17:21:11 2022

@author: anton
"""

import numpy as np
from scipy.stats import laplace
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.stats import  kurtosis
from scipy.stats import skew


def spectral_kurtosis(envelop_signal,fmin,fmax,atol = 0.15):
    
    assert len(envelop_signal) >0 , "Il faut avoir un signal enveloppe !"
    enveloppe = envelop_signal
    enveloppe_to_focus=enveloppe[enveloppe['fréquence (Hz)']>fmin]
    enveloppe_to_focus=enveloppe[enveloppe['fréquence (Hz)']<fmax]

    pic = enveloppe_to_focus['fréquence (Hz)'][enveloppe_to_focus['Amplitude'] == enveloppe_to_focus['Amplitude'].max()].iloc[0]
    enveloppe=enveloppe[enveloppe['fréquence (Hz)']>pic-0.5]
    enveloppe=enveloppe[enveloppe['fréquence (Hz)']<pic+0.5]

    #Aperçu des fréquences évaluées :
    enveloppe.plot(x ='fréquence (Hz)' ,y = "Amplitude",title = "Zoom sur le spectre centré en %f"%pic, kind = 'bar')
    
    enveloppe['next_value'] = enveloppe['Amplitude'].shift(periods = -1, fill_value = 0) #Décale la colonne des Hz vers le haut et arrondi par excés
    enveloppe['previous_value'] = enveloppe['Amplitude'].shift(periods = 1, fill_value = 0) #Décale la colonne des Hz vers le haut et arrondi par excés
    
    m = []
    M = []
    enveloppe['is_ok'] = enveloppe.apply(lambda x: is_ok(M,m,pic,x['fréquence (Hz)'],x['Amplitude'],x['next_value'],x['previous_value']),axis = 1)
    
    #On retire les pics qui empêcherai de calculer correctement le kurtosis
    enveloppe = enveloppe[enveloppe['is_ok'] == True]
    
    mean = enveloppe['fréquence (Hz)'][enveloppe['Amplitude'] > enveloppe['Amplitude'].mean()].iloc[0]
    enveloppe['deviation'] = enveloppe.apply(lambda x: x['Amplitude']*(x['fréquence (Hz)'] - pic)**2,axis =1)
    enveloppe['deviation_scale'] = enveloppe.apply(lambda x: x['Amplitude']*abs((x['fréquence (Hz)'] - mean)),axis =1)
    
    enveloppe.plot(x ='fréquence (Hz)' ,y = "Amplitude",title = "Zoom sur le spectre arrangé en %f"%pic, kind= 'bar')

    b = 1/sum(enveloppe['Amplitude'])*sum(enveloppe['deviation_scale'])
    
    #Distribution laplacienne
    dist = laplace(pic,b)
    hist = enveloppe['Amplitude']
    
    
    #On crée la courbe de densité de probabilité réelle
    prob = pd.DataFrame(hist/sum(enveloppe['Amplitude']) , columns = ['Amplitude'])
    prob['fréquence (Hz)'] = enveloppe['fréquence (Hz)']
    
    #Quelques tests#
    #Est-ce une prob ?
    # assert prob.sum() == 1 , "Ce n'est pas une probabilité"
    

    prob['cdf'] = prob['Amplitude'].cumsum()
    prob['cdf_approximé'] = dist.cdf(prob['fréquence (Hz)'])
    
    #Affichage#
    prob.plot(x ='fréquence (Hz)' ,y = ['cdf','cdf_approximé'],title = "CDF de l'approximation normale et du signal réel")
    
    
    # We consider the distribution is normal , plotting of the normal approximation
    fig , ax = plt.subplots()
    
    enveloppe['approximation_normale'] = enveloppe['fréquence (Hz)'].apply(lambda x: dist.pdf(x))
    
    #Remise à échelle :
    pace = 1/sum(enveloppe['approximation_normale'])
    M= np.arange(pic-0.5,pic+0.5,0.01)
    distribution = dist.pdf(M)*pace
    
    ax.plot(M,distribution,label='densité laplacienne continue')
    ax.plot(prob['fréquence (Hz)'],prob['Amplitude'], label = 'probabilité discrétisée')
    
    #Remise à niveau
    distribution=distribution*sum(enveloppe['Amplitude'])
    ecart = distribution.max() - hist.max()  
    distribution += -ecart
    #Affichage
    fig , ax2 = plt.subplots()

    ax2.plot(M,distribution,label='approximation laplacienne continue')
    ax2.plot(prob['fréquence (Hz)'],hist, label = 'spectre discrétisée')
    ax2.legend()
    ax2.set_title("Densités de probabilité du signal")
    
    #Précis ? Si ça ne l'est pas , la distribution n'est pas laplacienne et est très suceptible de ne pas être un défaut.
    index=prob[prob['fréquence (Hz)']==pic].index[0]
    prob=prob.loc[index-1:index+1]
    print(prob)
    assert len(prob) >=3, "ce n'est pas un pic"

    M= prob['fréquence (Hz)'].to_numpy()
    distribution=dist.pdf(M)*pace
    ecart = distribution.max() - prob['Amplitude'].max()  
    distribution += -ecart
    
    
    assert np.allclose(distribution , prob['Amplitude'],rtol = 0 , atol = atol) == True , "Approximation erronée à %f "%(atol*100) +'%'

    #Evaluation du kurtosis
    data = dist.rvs(size=1000000) #Big size to get the decrease random influence.
    K = kurtosis(data, fisher=True)
    Sk = skew(data)
    print(Sk)
    return enveloppe
    if( K > 2.5) :
        print("Signal impulsif de kurtosis %f: c'est un défaut" %K)
        return True
    else :
        print("Signal non impulsif de kurtosis %f : ce n'est pas un défaut" %K)
        return False
    
def is_ok(M,m,pic,current_fq,current_amp,next_amp,previous_amp):
    
    if (current_fq< pic):
        if (current_amp < next_amp):
            if (len(M) > 0): 
                if (current_amp > max(M)):
                    M.append(current_amp)
                    return True
                else : return False
            else :
                M.append(current_amp)
                return True
        else :
            return False
        
    elif (current_fq > pic):
        if (previous_amp > current_amp):
            if (len(m) > 0): 
                if (current_amp < min(m)):
                    m.append(current_amp)
                    return True
                else : return False
            else :
                m.append(current_amp)
                return True
        else :
            return False
    elif (current_fq == pic):
        return True