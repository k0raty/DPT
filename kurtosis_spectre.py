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


def spectral_kurtosis(sample,fmin,fmax):
    
    sample=sample[sample['fréquence (Hz)']>fmin]
    sample=sample[sample['fréquence (Hz)']<fmax]

    mean = sample['fréquence (Hz)'][sample['Amplitude'] == sample['Amplitude'].max()].iloc[0]
    
    #Aperçu des fréquences évaluées :
    sample.plot(x ='fréquence (Hz)' ,y = "Amplitude",title = "Zoom sur le spectre centré en %f"%mean, kind = 'bar')
    
    sample['next_value'] = sample['Amplitude'].shift(periods = -1, fill_value = 0) #Décale la colonne des Hz vers le haut et arrondi par excés
    sample['previous_value'] = sample['Amplitude'].shift(periods = 1, fill_value = 0) #Décale la colonne des Hz vers le haut et arrondi par excés
    m = []
    M = []
    sample['is_ok'] = sample.apply(lambda x: is_ok(M,m,mean,x['fréquence (Hz)'],x['Amplitude'],x['next_value'],x['previous_value']),axis = 1)
    sample['deviation'] = sample.apply(lambda x: x['Amplitude']*(x['fréquence (Hz)'] - mean)**2,axis =1)
    sample['deviation_scale'] = sample.apply(lambda x: x['Amplitude']*abs((x['fréquence (Hz)'] - mean)),axis =1)
    
    #On retire les pics qui empêcherai de calculer correctement le kurtosis
    sample = sample[sample['is_ok'] == True]
    sample.plot(x ='fréquence (Hz)' ,y = "Amplitude",title = "Zoom sur le spectre arrangé en %f"%mean, kind= 'bar')

    b = 1/sum(sample['Amplitude'])*sum(sample['deviation_scale'])
    
    dist = laplace(mean,b)
    hist = sample['Amplitude']
    
    
    #On crée la courbe de densité de probabilité réelle
    prob = pd.DataFrame(hist/sum(sample['Amplitude']) , columns = ['Amplitude'])
    prob['fréquence (Hz)'] = sample['fréquence (Hz)']
    
    #Quelques tests#
    #Est-ce une prob ?
    # assert prob.sum() == 1 , "Ce n'est pas une probabilité"
    
    #Précis ?

    prob['cdf'] = prob['Amplitude'].cumsum()
    prob['cdf_approximé'] = dist.cdf(prob['fréquence (Hz)'])
#    assert np.allclose(prob['cdf'] , prob['cdf_approximé'],rtol = 0 , atol = 0.05) == True , "Approximation erronée à 5 %"
    
    #Affichage#
    prob.plot(x ='fréquence (Hz)' ,y = ['cdf','cdf_approximé'],title = "CDF de l'approximation normale et du signal réel")
    
    # We consider the distribution is normal , plotting of the normal approximation
    fig , ax = plt.subplots()
    
    sample['approximation_normale'] = sample['fréquence (Hz)'].apply(lambda x: dist.pdf(x))
    
    #Remise à échelle :
    pace = 1/sum(sample['approximation_normale'])
    M= np.arange(fmin,fmax,0.01)
    distribution = dist.pdf(M)*pace
    sample['approximation_normale'] = sample['approximation_normale']*pace
    
    #Affichage
    ax.plot(M,distribution,label='approximation normale continue')
    ax.plot(prob['fréquence (Hz)'],prob['Amplitude'], label = 'probabilité discrétisée')
    ax.legend()
    ax.set_title("Densités de probabilité du signal")
    
    #Evaluation du kurtosis
    data = dist.rvs(size=1000000) #Big size to get the decrease random influence.
    K = kurtosis(data, fisher=True)
    if( K > 2.5) :
        print("Signal impulsif de kurtosis %f: c'est un défaut" %K)
        return True
    else :
        print("Signal non impulsif de kurtosis %f : ce n'est pas un défaut" %K)
        return False
def is_ok(M,m,mean,current_fq,current_amp,next_amp,previous_amp):
    
    if (current_fq< mean):
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
        
    elif (current_fq > mean):
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
    elif (current_fq == mean):
        return True
