import matplotlib.pyplot as plt
import numpy as np

f = 5
f_s = 100

t = np.linspace(0, 2, 2*f_s, endpoint=False)
x = .5*np.sin(f*2*np.pi*t) + np.sin(2*f*2*np.pi*t) \
	 + np.cos(f*2*np.pi*t) + .5*np.cos(2*f*2*np.pi*t)

fig, ax = plt.subplots()
ax.plot(t, x)
ax.set_xlabel('Time [s]')
ax.set_ylabel('Signal amplitude')
plt.show()

from scipy import fftpack

X = fftpack.fft(x)
freqs = fftpack.fftfreq(len(x)) * f_s

X = np.abs(X)
# reduce all non peaks to 0
X[X < 1e-10] = 0.0

# get all non-zero frequencies
print(X)
peak_freqs = list(set(np.abs(freqs[np.nonzero(X)])))
print(peak_freqs)

fig, ax = plt.subplots()

ax.stem(freqs, np.abs(X))
ax.set_xlabel('Frequency in Hertz [Hz]')
ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
ax.set_xlim(-f_s / 2, f_s / 2)
ax.set_ylim(-5, 110)
plt.show()