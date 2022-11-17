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
from scipy.stats import norm



def spectral_kurtosis(envelop_signal,fmin,fmax,atol = 0.07,seuil_laplace = 10e-7,seuil_norm =50000):
    """
    curvature of a function : https://tutorial.math.lamar.edu/classes/calciii/curvature.aspx
    """
    
    assert len(envelop_signal) >0 , "Il faut avoir un signal enveloppe !"
    
    enveloppe = envelop_signal
    enveloppe_to_focus=enveloppe[enveloppe['fréquence (Hz)']>fmin]
    enveloppe_to_focus=enveloppe_to_focus[enveloppe_to_focus['fréquence (Hz)']<fmax]
    pic = enveloppe_to_focus['fréquence (Hz)'][enveloppe_to_focus['Amplitude'] == enveloppe_to_focus['Amplitude'].max()].iloc[0]
    del(enveloppe_to_focus)
    enveloppe=enveloppe[enveloppe['fréquence (Hz)']>pic-0.5]
    enveloppe=enveloppe[enveloppe['fréquence (Hz)']<pic+0.5]

    #Aperçu des fréquences évaluées :
    enveloppe.plot(x ='fréquence (Hz)' ,y = "Amplitude",title = "Zoom sur le spectre centré en %f"%pic, kind = 'bar')
    
    enveloppe['next_value'] = enveloppe['Amplitude'].shift(periods = -1, fill_value = 0) #Décale la colonne des Hz vers le haut et arrondi par excés
    enveloppe['previous_value'] = enveloppe['Amplitude'].shift(periods = 1, fill_value = 0) #Décale la colonne des Hz vers le haut et arrondi par excés
    
    m = []
    M = []
    max_amp = enveloppe['Amplitude'].max()
    enveloppe['is_ok'] = enveloppe.apply(lambda x: is_ok(M,m,pic,x['fréquence (Hz)'],x['Amplitude'],x['next_value'],x['previous_value'],max_amp),axis = 1)
    
    #On retire les pics qui empêcherai de calculer correctement le kurtosis
    enveloppe = enveloppe[enveloppe['is_ok'] == True]
    
    #On retire ceux qui seraient passé inaperçu
    current_index=enveloppe[enveloppe['fréquence (Hz)']==pic].index[0]
    current_index = list(enveloppe.index).index(current_index)
    
    #A gauche
    before_pic = enveloppe.iloc[:current_index]
    before_min = before_pic['Amplitude'].min()
    before_min_index = before_pic[before_pic['Amplitude'] == before_min].index[0]
    before_min_index = list(enveloppe.index).index(before_min_index)
    
    enveloppe = enveloppe.iloc[before_min_index:]
    
    #A droite 
    current_index=enveloppe[enveloppe['fréquence (Hz)']==pic].index[0]
    current_index = list(enveloppe.index).index(current_index)
    
    after_pic = enveloppe.iloc[current_index+1:]
    after_max = after_pic['Amplitude'].max()
    after_max_index = after_pic[after_pic['Amplitude'] == after_max].index[0]
    after_max_index = list(enveloppe.index).index(after_max_index)
    
    if after_max_index > current_index+1:
        enveloppe = enveloppe.iloc[:after_max_index]
    #On centre sur la valeur principal si il y a plus de 4 valeurs :

    #On identifie ses voisins
    if len(enveloppe) > 4:
        length_inf = len(enveloppe.index[:current_index])
        length_sup = len(enveloppe.index[current_index+1:])
        length = abs(length_sup-length_inf)
        
        if length_inf > length_sup:
            enveloppe=enveloppe.iloc[length:]
        elif length_inf < length_sup:
            enveloppe=enveloppe.iloc[:-length]
            
        assert len(enveloppe) >=3, "ce n'est pas un pic"
    
    enveloppe['mean'] = enveloppe.apply(lambda x:x['Amplitude']*x['fréquence (Hz)'],axis =1) #Le poids de chaque 

    mean = sum(enveloppe['mean'])/sum(enveloppe['Amplitude'])
    
    enveloppe['deviation'] = enveloppe.apply(lambda x: x['Amplitude']*(x['fréquence (Hz)'] - mean)**2,axis =1)
    enveloppe['deviation_scale'] = enveloppe.apply(lambda x: x['Amplitude']*abs((x['fréquence (Hz)'] - mean)),axis =1) #Le poids de chaque 

    enveloppe.plot(x ='fréquence (Hz)' ,y = "Amplitude",title = "Zoom sur le spectre arrangé en %f"%pic, kind= 'bar')

    b = (1/sum(enveloppe['Amplitude']))*sum(enveloppe['deviation_scale'])
    std = np.sqrt((1/sum(enveloppe['Amplitude']))*sum(enveloppe['deviation']))
    #Distribution laplacienne
    dist = laplace(pic,b)
    dist_norm = norm(pic,std)
    hist = enveloppe['Amplitude']
    
    
    #On crée la courbe de densité de probabilité réelle
    prob = pd.DataFrame(hist/sum(enveloppe['Amplitude']) , columns = ['Amplitude'])
    prob['fréquence (Hz)'] = enveloppe['fréquence (Hz)']
    
    #Quelques tests#
    #Est-ce une prob ?
    # assert prob.sum() == 1 , "Ce n'est pas une probabilité"
    

    prob['cdf'] = prob['Amplitude'].cumsum()
    prob['cdf_approximé'] = dist.cdf(prob['fréquence (Hz)'])
    prob['cdf_approximé_norm'] = dist_norm.cdf(prob['fréquence (Hz)'])
    
    #Affichage#
    prob.plot(x ='fréquence (Hz)' ,y = ['cdf','cdf_approximé'],title = "CDF de l'approximation normale et du signal réel")
    
    
    # We consider the distribution is normal , plotting of the normal approximation
    fig , ax = plt.subplots()
    
    enveloppe['approximation_laplace'] = enveloppe['fréquence (Hz)'].apply(lambda x: dist.pdf(x))
    enveloppe['approximation_normale'] = enveloppe['fréquence (Hz)'].apply(lambda x: dist_norm.pdf(x))

    #Remise à échelle :
    pace = 1/sum(enveloppe['approximation_laplace'])
    pace_norm = 1/sum(enveloppe['approximation_normale'])
    M= np.arange(pic-0.5,pic+0.5,0.01)
    distribution = dist.pdf(M)*pace
    distribution_norm = dist_norm.pdf(M)*pace_norm

    
    ax.plot(M,distribution,label='densité laplacienne continue')
    ax.plot(M,distribution_norm,label='densité normale continue')

    ax.plot(prob['fréquence (Hz)'],prob['Amplitude'], label = 'probabilité discrétisée')
    
    #Remise à niveau
    #Laplace
    distribution=distribution*sum(enveloppe['Amplitude'])
    ecart = distribution.max() - hist.max()  
    distribution += -ecart
    #Gauss
    distribution_norm=distribution_norm*sum(enveloppe['Amplitude'])
    ecart = distribution_norm.max() - hist.max()  
    distribution_norm += -ecart
    
    #Affichage
    fig , ax2 = plt.subplots()

    ax2.plot(M,distribution,label='approximation laplacienne continue')
    ax2.plot(M,distribution_norm,label='approximation normale continue')
    ax2.plot(prob['fréquence (Hz)'],hist, label = 'spectre discrétisée')
    ax2.legend()
    ax2.set_title("Densités de probabilité du signal")
    #Précis ? Si ça ne l'est pas , la distribution n'est pas laplacienne et est très suceptible de ne pas être un défaut.
    
    #On cherche l'index de la fréquence cummulante
    try : #Try car il peut y avoir des configuration ou cela ne fonctionne pas ( le pic est à l'extremité ect ... on considère alors que cela est faux)
        current_index=prob[prob['fréquence (Hz)']==pic].index[0]
        current_index = list(prob.index).index(current_index)
        
        #On identifie ses voisins
        index_inf = prob.index[current_index-1]
        index_sup = prob.index[current_index+1]
        prob=prob.loc[index_inf:index_sup]
        assert len(prob) >=3, "ce n'est pas un pic"
    
        #Laplace
        M= prob['fréquence (Hz)'].to_numpy()
        distribution=dist.pdf(M)*pace
        ecart = distribution.max() - prob['Amplitude'].max()  
        distribution += -ecart
        
        #Normale
        distribution_norm=dist_norm.pdf(M)*pace
        ecart = distribution_norm.max() - prob['Amplitude'].max()  
        distribution_norm += -ecart
    except : 
        print("Ce n'est pas un pic")
        return False
    
    
    if (np.allclose(distribution , prob['Amplitude'],rtol = 0 , atol = atol) == False and np.allclose(distribution_norm , prob['Amplitude'],rtol = 0 , atol = atol) == False): #Appel récursif
        print("Approximation erronée à %f "%(atol*100) +'%')
        if (len(enveloppe) == 5) : #Il peut arriver que la fréquence de juste à côté soit quasiment au même niveau que le pic , on la retire.
            thresh = enveloppe['Amplitude'].max()
            envelop_signal = enveloppe[enveloppe['Amplitude'] < thresh/2]
            envelop_signal=envelop_signal.append(enveloppe[enveloppe['Amplitude'] == thresh])
            envelop_signal=envelop_signal.sort_values(by = ['fréquence (Hz)'])
        else : envelop_signal = enveloppe.iloc[1:-1]
        if (len(envelop_signal) < 3):
            print("Ce n'est pas un pic")
            return False
        return spectral_kurtosis(envelop_signal,fmin,fmax,atol)
    else :
        #Evaluation de la courbure / kurtosis :f''/((1+f'**2)**(3/2))
        #data = dist.rvs(size=1000000) #Big size to get the decrease random influence.
        #K = kurtosis(data, fisher=True)
        #Sk = skew(data)
        if (np.allclose(distribution , prob['Amplitude'],rtol = 0 , atol = atol) == True):
            D = sum(enveloppe['Amplitude'])*pace
            Kappa = D/(2*b**3*(1+D**2/(4*b**4))**(3/2))
            if( Kappa < seuil_laplace) :
                print("Signal impulsif laplacien : c'est un défaut")
                print("Kappa : ", Kappa)
                return True
            else :
                print("Signal non impulsif laplacien: ce n'est pas un défaut" %Kappa)
                print("Kappa : ", Kappa)
                return False
        else :
            D = sum(enveloppe['Amplitude'])*pace_norm
            Kappa = D/(np.sqrt(2*np.pi)*std**3)
            if( Kappa > seuil_norm) :
                print("Signal impulsif normale: c'est un défaut")
                print("Kappa : ", Kappa)
                return True
            else :
                print("Signal non impulsif normale : ce n'est pas un défaut" %Kappa)
                print("Kappa : ", Kappa)
                return False
    
        
def is_ok(M,m,pic,current_fq,current_amp,next_amp,previous_amp,max_amp):
    
    #On conserve tout de même les fréquences voisines pour éviter un déséquilibrage dans le pas entre chaque fréquence
    if (next_amp == max_amp):
        return True
    if (previous_amp == max_amp):
        return True 
    if (current_fq< pic):
        if (current_amp < next_amp):
            if (len(M) > 0): 
                if (current_amp > max(M)):
                    M.append(current_amp)
                    return True
                elif (current_amp < min(M)):
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