import numpy as np
from scipy.fftpack import fft,ifft
from scipy import signal
import matplotlib.pyplot as plt
import seaborn

def PLOT(x):
    plt.plot(x)
    plt.show()

def FFT(x):
    xf = abs(fft(x))/len(x)
    xf = xf[range(int(len(xf)/2))]
    return xf

# 原始信号
fs = 1000
x=np.linspace(0,1,fs)
y=2*np.sin(2*np.pi*50*x)+5*np.sin(2*np.pi*100*x)
PLOT(y)

# 频域
yf = FFT(y)
PLOT(yf)

# 低通滤波
fc = 70
wn = 2*fc/fs
b, a = signal.butter(8, wn, 'lowpass')
y2 = signal.filtfilt(b, a, y)
PLOT(y2)
yf2 = FFT(y2)
PLOT(yf2)