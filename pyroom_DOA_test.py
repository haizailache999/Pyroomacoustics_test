import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve

import pyroomacoustics as pra
from pyroomacoustics.doa import circ_dist
import warnings
warnings.filterwarnings("ignore")

######
# We define a meaningful distance measure on the circle

# Location of original source
azimuth = 68.0 / 180.0 * np.pi  # 60 degrees
azimuth2 = 111.0 / 180.0 * np.pi  # 60 degrees
azimuth3 = 178.0 / 180.0 * np.pi  # 60 degrees
azimuth4 = 221.0 / 180.0 * np.pi  # 60 degrees
azimuth5 = 332.0 / 180.0 * np.pi  # 60 degrees
azimuth_group=[azimuth,azimuth2,azimuth3,azimuth4,azimuth5]
distance = 3.0  # 3 meters
dim = 2  # dimensions (2 or 3)
room_dim = np.r_[10.0, 10.0]

# Use AnechoicRoom or ShoeBox implementation. The results are equivalent because max_order=0 for both.
# The plots change a little because in one case there are no walls.
use_anechoic_class = False

#print("============ Using anechoic: {} ==================".format(anechoic))

#######################
# algorithms parameters
SNR = 0.0  # signal-to-noise ratio
c = 343.0  # speed of sound
fs = 16000  # sampling frequency
nfft = 256  # FFT size
freq_bins = np.arange(5, 60)  # FFT bins to use for estimation

# compute the noise variance
sigma2 = 10 ** (-SNR / 10) / (4.0 * np.pi * distance) ** 2

# Create an anechoic room
if use_anechoic_class:
    aroom = pra.AnechoicRoom(dim, fs=fs, sigma2_awgn=sigma2)
else:
    aroom = pra.ShoeBox(room_dim, fs=fs, max_order=0, sigma2_awgn=sigma2)

# add the source
source_location = [room_dim / 2 + distance * np.r_[np.cos(azimuth), np.sin(azimuth)] for azimuth in azimuth_group]
#print(source_location)
source_signal = np.random.randn((nfft // 2 + 1) * nfft)
#print(source_signal.shape)
for i in range(len(source_location)):
    aroom.add_source(source_location[i], signal=source_signal)

# We use a circular array with radius 15 cm # and 12 microphones
R = pra.circular_2D_array(room_dim / 2, 12, 0.0, 0.15)
aroom.add_microphone_array(pra.MicrophoneArray(R, fs=aroom.fs))

# run the simulation
aroom.simulate()

#for signal in aroom.mic_array.signals:
    #print(aroom.mic_array.signals.shape)
################################
# Compute the STFT frames needed
X = np.array(
    [
        pra.transform.stft.analysis(signal, nfft, nfft // 2).T
        for signal in aroom.mic_array.signals
    ]
)

#print(X.shape)

##############################################
# Now we can test all the algorithms available
algo_names = sorted(pra.doa.algorithms.keys())
#print(algo_names)

for algo_name in algo_names:
    # Construct the new DOA object
    # the max_four parameter is necessary for FRIDA only
    doa = pra.doa.algorithms[algo_name](R, fs, nfft, c=c, num_src=5,max_four=10)

    # this call here perform localization on the frames in X
    doa.locate_sources(X)
    #doa.grid.plot(doa.P/np.max(doa.P))
    #doa.polar_plt_dirac()
    plt.title(algo_name)

    # doa.azimuth_recon contains the reconstructed location of the source
    print(algo_name)
    print("  Recovered azimuth:", doa.azimuth_recon / np.pi * 180.0, "degrees")
    print("  Error:", circ_dist(azimuth_group, doa.azimuth_recon) / np.pi * 180.0, "degrees")

#plt.show()