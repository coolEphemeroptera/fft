import numpy as np
from scipy.fftpack import fft,ifft
from scipy import signal
import matplotlib.pyplot as plt


def PLOT(y,x=None):
    if x is None:
        x = np.arange(0,y.shape[0])
    plt.plot(x,y)
    plt.show()

def FFT(x):
    xf = abs(fft(x))/len(x)
    xf = xf[range(int(len(xf)/2))]
    return xf

# 原始信号
# 采样频率
fs = 1000
# 采样点0~2s
t=np.linspace(0,2,2*fs)
# f1=50hz 谐波和 f2=100hz 谐波
f1 = 50
f2 = 100
# theta = wt = 2πft = 2πfn/fs
theta1 = 2*np.pi*f1*t
theta2 = 2*np.pi*f2*t
# 谐波叠加
y=2*np.sin(theta1)+5*np.sin(theta2)
PLOT(y,t)

# 傅里叶变换
# 频率坐标
f = fs*np.arange(0,int(y.shape[0]/2))/y.shape[0]
yf = FFT(y)
PLOT(yf,f)

# 低通滤波
fc = 70
wn = 2*fc/fs
b, a = signal.butter(8, wn, 'lowpass')
y2 = signal.filtfilt(b, a, y)
PLOT(y2,t)
yf2 = FFT(y2)
PLOT(yf2,f)