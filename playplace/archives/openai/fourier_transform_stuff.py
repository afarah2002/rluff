# import matplotlib.pyplot as plt
# import numpy as np

# f = 5
# f_s = 100

# t = np.linspace(0, 2, 2*f_s, endpoint=False)
# x = .5*np.sin(f*2*np.pi*t) + np.sin(2*f*2*np.pi*t) \
# 	 + np.cos(f*2*np.pi*t) + .5*np.cos(2*f*2*np.pi*t)

# fig, ax = plt.subplots()
# ax.plot(t, x)
# ax.set_xlabel('Time [s]')
# ax.set_ylabel('Signal amplitude')
# plt.show()

# from scipy import fftpack

# X = fftpack.fft(x)
# freqs = fftpack.fftfreq(len(x)) * f_s

# X = np.abs(X)
# # reduce all non peaks to 0
# X[X < 1e-10] = 0.0

# # get all non-zero frequencies
# print(X)
# peak_freqs = list(set(np.abs(freqs[np.nonzero(X)])))
# print(peak_freqs)



import numpy as np
import matplotlib.pyplot as plt

# Time period
t = np.arange(0, 10, 0.01);
# Create a sine wave with multiple frequencies(1 Hz, 2 Hz and 4 Hz)
a = np.sin(2*np.pi*t) + np.sin(2*2*np.pi*t) + np.sin(4*2*np.pi*t);
# Do a Fourier transform on the signal
tx  = np.fft.fft(a);
# Do an inverse Fourier transform on the signal
itx = np.fft.ifft(tx);
# Plot the original sine wave using inverse Fourier transform
plt.plot(t, itx);
plt.title("Sine wave plotted using inverse Fourier transform");
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show();