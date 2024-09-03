import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
import math

np.random.seed(0)
fs = 16000  
signal_length = 47520  
room_dim = [8, 9]  

room = pra.ShoeBox(room_dim, fs=fs, absorption=1.0, max_order=1)

mic_positions = np.array([[5.8,5.9,6],  
                          [6,6,6]])  
mic_array = pra.MicrophoneArray(mic_positions, fs)
room.add_microphone_array(mic_array)

source_position = [2, 7]  
source_signal = np.random.randn(signal_length)  
room.add_source(source_position, signal=source_signal)

room.compute_rir()

received_signals = room.simulate(return_premix=True)
received_signals=np.squeeze(received_signals)
X = np.array(
    [
        pra.transform.stft.analysis(signal, 256, 256 // 2).T
        for signal in received_signals
    ]
)
print(X.shape)

doa = pra.doa.music.MUSIC(mic_array.R, fs, nfft=256, num_src=1)  
doa.locate_sources(X)  
print(doa.azimuth_recon / np.pi * 180.0)
print("upper bound",math.atan2(5.8-2,6-7)*180/math.pi)
print("lower bound",math.atan2(6-2,6-7)*180/math.pi)
print("medium bound",math.atan2(5.9-2,6-7)*180/math.pi)


room.plot(img_order=0)
source_position_plot=np.array([source_position]).T
plt.scatter(source_position_plot[0,:],source_position_plot[1, :], c='r', marker='x', label='Signal Sources')
plt.legend()
plt.show()