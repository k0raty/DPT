"""
https://pywavelets.readthedocs.io/en/latest/ref/cwt.html
http://www.pyrunner.com/weblog/2016/08/01/optimal-svht/
call mathlab function : 
    https://fr.mathworks.com/help/matlab/matlab_external/call-matlab-functions-from-python.html
    inverse cwt : https://fr.mathworks.com/help/wavelet/gs/inverse-continuous-wavelet-transform.html
    reconstruct the signal from a cwt process : 
        https://fr.mathworks.com/help/wavelet/ug/signal-reconstruction-from-continuous-wavelet-transform-coefficients.html
inverse dwt: 
    https://pywavelets.readthedocs.io/en/latest/ref/idwt-inverse-discrete-wavelet-transform.html
find threshold : thresholds are only for dwt...it is an approx to focus on frequencies (the lower the better)
    file:///C:/Users/anton/Downloads/machines-10-00649.pdf
details coefficient and threshold : 
    https://www.sciencedirect.com/topics/engineering/detail-coefficient
"""
import numpy as np

import pywt
import matplotlib.pyplot as plt
import pandas as pd



fs = 50000*2

dt = 1/fs

frequencies = np.array([8.333, 63.731, 31.865, 93.663, 72.964, 3.65, 4.683])  # normalize

 
def frequency2scale(frequence):
    """
    return frequencies relative to scales for a wavelet
    """
    Fc = pywt.central_frequency('cmor1.5-1.0', precision=8)
  #  Fc = eng.centfrq('cmor1.5-1.0')
    return Fc/(frequence*dt)

scales=list(map(frequency2scale,frequencies))
assert any(frequencies.round(3) == (pywt.scale2frequency('cmor1.5-1.0' ,  scales)*10**5).round(3)) == True , "Problème sur les fréquences"
signal = pd.read_csv('signal_temporel_filtré.csv',sep=',')
coef, freqs = pywt.cwt(signal['signal_moyen'].to_numpy(),scales ,'cmor1.5-1.0', sampling_period = dt)


def valeur_seuillage(coef): #Appliquer un map
    """
    Le théorème de Donoho et Johnstone définit un seuil T pour les coefficients de la
transformée en ondelettes
    22ème Congrès Français de Mécanique
    return Threshold to cut coef from its noise
    """
    norm_coef = np.abs(coef)
    med = np.median(norm_coef)
    sigma = med / 0.6745
    T = sigma*np.sqrt(2*np.log(len(coef)))
    
    return T
    
thresholds = list(map(valeur_seuillage,coef))
def cut(coef,threshold):
    return coef[np.abs(coef)>threshold]
def cut_2(coef,threshold):
    return pywt.threshold(coef, threshold, mode='hard')
coef_cut = list(map(cut,coef,thresholds))


"""
# print the range over which the wavelet will be evaluated
print("Continuous wavelet will be evaluated over the range [{}, {}]".format(
    wav.lower_bound, wav.upper_bound))

width = wav.upper_bound - wav.lower_bound


max_len = int(np.max(scales)*width + 1)
t = np.arange(max_len)
fig, axes = plt.subplots(len(scales), 2, figsize=(12, 6))

for n, scale in enumerate(scales):

    # The following code is adapted from the internals of cwt
    int_psi, x = pywt.integrate_wavelet(wav, precision=10)
    step = x[1] - x[0]
    j = np.floor(
        np.arange(scale * width + 1) / (scale * step))
    if np.max(j) >= np.size(int_psi):
        j = np.delete(j, np.where((j >= np.size(int_psi)))[0])
    j = j.astype(np.int_)

    # normalize int_psi for easier plotting
    int_psi /= np.abs(int_psi).max()

    # discrete samples of the integrated wavelet
    filt = int_psi[j][::-1]

    # The CWT consists of convolution of filt with the signal at this scale
    # Here we plot this discrete convolution kernel at each scale.

    nt = len(filt)
    t = np.linspace(-nt//2, nt//2, nt)
    axes[n, 0].plot(t, filt.real, t, filt.imag)
    axes[n, 0].set_xlim([-max_len//2, max_len//2])
    axes[n, 0].set_ylim([-1, 1])
    axes[n, 0].text(50, 0.35, 'scale = {}'.format(scale))

    f = np.linspace(-np.pi, np.pi, max_len)
    filt_fft = np.fft.fftshift(np.fft.fft(filt, n=max_len))
    filt_fft /= np.abs(filt_fft).max()
    axes[n, 1].plot(f, np.abs(filt_fft)**2)
    axes[n, 1].set_xlim([-np.pi, np.pi])
    axes[n, 1].set_ylim([0, 1])
    axes[n, 1].set_xticks([-np.pi, 0, np.pi])
    axes[n, 1].set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
    axes[n, 1].grid(True, axis='x')
    axes[n, 1].text(np.pi/2, 0.5, 'scale = {}'.format(scale))

axes[n, 0].set_xlabel('time (samples)')
axes[n, 1].set_xlabel('frequency (radians)')
axes[0, 0].legend(['real', 'imaginary'], loc='upper left')
axes[0, 1].legend(['Power'], loc='upper left')
axes[0, 0].set_title('filter')
axes[0, 1].set_title(r'|FFT(filter)|$^2$')
"""