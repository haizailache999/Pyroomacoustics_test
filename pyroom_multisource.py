import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import soundfile as sf

np.random.seed(0)
room_dim = [10, 8]  
room = pra.ShoeBox(room_dim, fs=16000, max_order=10, absorption=0.5)  

source_positions = np.array([
    [3,5],
    [7,2],
    [8,6],
    [6,7],
    [4,3]
]).T
num_sources = source_positions.shape[1]  
for i in range(num_sources):
    room.add_source(source_positions[:, i])

mic_positions = np.array([[6, 6.1, 6.1],   
                          [4, 3.8, 3.6]])  

mic_array = pra.MicrophoneArray(mic_positions, room.fs)
room.add_microphone_array(mic_array)

room.plot(img_order=0)
plt.scatter(source_positions[0, :], source_positions[1, :], c='r', marker='x', label='Signal Sources')
plt.legend()
plt.show()

room.compute_rir()

signal_duration = 2  
num_samples = int(signal_duration * room.fs)

flac_filenames=["C:/Users/yifengw3/Desktop/yifeng_project1/dataset/61-70968-0001.flac","C:/Users/yifengw3/Desktop/yifeng_project1/dataset/61-70968-0002.flac","C:/Users/yifengw3/Desktop/yifeng_project1/dataset/61-70968-0003.flac","C:/Users/yifengw3/Desktop/yifeng_project1/dataset/61-70968-0004.flac","C:/Users/yifengw3/Desktop/yifeng_project1/dataset/61-70968-0005.flac"]
source_signals=[]
for flac_file in flac_filenames:
    audio,fs=sf.read(flac_file)
    #print(fs)
    if fs!=room.fs:
        raise ValueError("wrong signal fs!")
    audio=audio.astype(np.float32)/np.max(np.abs(audio))
    #print(audio.shape)
    source_signals.append(audio)

num_samples=min([len(sig) for sig in source_signals])
target_signal=[sig[:num_samples] for sig in source_signals]
#target_signal = np.random.randn(num_samples)
#target_signal_power=np.mean(target_signal[0]**2)
print("signal power: ",[np.mean(target_signal[i]**2) for i in range(len(target_signal))])
print("signal max amplitude",np.max(target_signal))
#print(target_signal)

received_signals = np.zeros((3, num_samples))
for mic in range(3):
    for i, source in enumerate(room.sources):
        rir=room.rir[mic][i]
        #print(target_signal.shape,rir.shape)
        mid=np.convolve(target_signal[i], rir[:num_samples], mode='same')
        #print(mid.shape)
        received_signals[mic, :] += mid

def calculate_si_snr(target, estimate):
    target = np.asarray(target).flatten()
    estimate = np.asarray(estimate).flatten()
    target_energy = np.sum(target ** 2)
    scale = np.dot(estimate, target) / target_energy
    s_target = scale * target
    e_noise = estimate - s_target
    si_snr = 10 * np.log10(np.sum(s_target ** 2) / np.sum(e_noise ** 2))
    return si_snr

misnr_values1 = [calculate_si_snr(target_signal[src], received_signals[0]) for src in range(5) ]
misnr_values2 = [calculate_si_snr(target_signal[src], received_signals[1]) for src in range(5) ]
misnr_values3 = [calculate_si_snr(target_signal[src], received_signals[2]) for src in range(5) ]
print(f'Si-snr for microphone1 (dB): {misnr_values1}')
print(f'Si-snr for microphone2 (dB): {misnr_values2}')
print(f'Si-snr for microphone3 (dB): {misnr_values3}')


def calculate_snr(signal, noise):
    power_signal = np.mean(signal ** 2)
    power_noise = np.mean(noise ** 2)
    snr = 10 * np.log10(power_signal / power_noise)
    return snr

noise_signal = np.random.randn(num_samples) * 1e-6  
snr_values = [calculate_snr(received_signals[mic], noise_signal) for mic in range(3)]

print(f'SNR for each microphone (dB) in Noise-Free Environment: {snr_values}')

def calculate_sdr(original, estimated):
    error_signal = original - estimated
    sdr = 10 * np.log10(np.sum(original ** 2) / np.sum(error_signal ** 2))
    return sdr

sdr_values1 = [calculate_sdr(target_signal[src], received_signals[0]) for src in range(5) ]
sdr_values2 = [calculate_sdr(target_signal[src], received_signals[1]) for src in range(5) ]
sdr_values3 = [calculate_sdr(target_signal[src], received_signals[2]) for src in range(5) ]
print(f'SDR for microphone1 (dB): {sdr_values1}')
print(f'SDR for microphone2 (dB): {sdr_values2}')
print(f'SDR for microphone3 (dB): {sdr_values3}')

def calculate_si_sdr(reference, estimated):
    alpha = np.dot(estimated, reference) / np.dot(reference, reference)
    projection = alpha * reference
    noise = estimated - projection
    si_sdr = 10 * np.log10(np.sum(projection ** 2) / np.sum(noise ** 2))
    return si_sdr

si_sdr_values0 = [calculate_si_sdr(target_signal[src], received_signals[0]) for src in range(5)]  
si_sdr_values1 = [calculate_si_sdr(target_signal[src], received_signals[1]) for src in range(5)]  
si_sdr_values2 = [calculate_si_sdr(target_signal[src], received_signals[2]) for src in range(5)]  

print(f'SI-SDR for microphone1 (dB): {si_sdr_values0}')
print(f'SI-SDR for microphone2 (dB): {si_sdr_values1}')
print(f'SI-SDR for microphone3 (dB): {si_sdr_values2}')


'''noise_level_db=-10
noise_std=10**(noise_level_db/20)
noise_signals=noise_std*np.random.randn(3,num_samples)
noise_signal_power=np.mean(noise_signals**2)'''
noise_file='C:/Users/yifengw3/Desktop/yifeng_project1/dataset/noise1.wav'
noise,fs=sf.read(noise_file)
#print(fs)
if fs!=room.fs:
    raise ValueError("Wrong noise fs!")
#print(noise.shape)
noise_total=np.mean(noise,axis=1)
noise=noise_total.astype(np.float32)/np.max(np.abs(noise_total))
if len(noise)<num_samples:
    repeats=int(np.ceil(num_samples/len(noise)))
    noise=np.tile(noise,repeats)[:num_samples]
else:
    noise=noise[:num_samples]
#print(noise.shape)
#print(num_samples)
noise_signals=np.vstack([noise]*3)
#print(noise_signals.shape)
print("************************************************")
print("noise power: ",[np.mean(noise_signals[i]**2) for i in range(len(noise_signals))])
print("Noise max amplitude",np.max(noise))

received_signals_noisy=received_signals+noise_signals

snr_values_noisy = [calculate_snr(received_signals_noisy[mic], noise_signals[mic]) for mic in range(3)]
print(f'SNR for each microphone (dB) in Noise Environment: {snr_values_noisy}')

sdr_values_noisy1 = [calculate_sdr(target_signal[src], received_signals_noisy[0]) for src in range(5) ]
sdr_values_noisy2 = [calculate_sdr(target_signal[src], received_signals_noisy[1]) for src in range(5) ]
sdr_values_noisy3 = [calculate_sdr(target_signal[src], received_signals_noisy[2]) for src in range(5) ]
print(f'SDR for microphone1 (dB) in Noise: {sdr_values_noisy1}')
print(f'SDR for microphone2 (dB) in Noise: {sdr_values_noisy2}')
print(f'SDR for microphone3 (dB) in Noise: {sdr_values_noisy3}')

si_sdr_values_noisy0 = [calculate_si_sdr(target_signal[src], received_signals_noisy[0]) for src in range(5)]  
si_sdr_values_noisy1 = [calculate_si_sdr(target_signal[src], received_signals_noisy[1]) for src in range(5)]  
si_sdr_values_noisy2 = [calculate_si_sdr(target_signal[src], received_signals_noisy[2]) for src in range(5)]  

print(f'SI-SDR for microphone1 (dB) in Noise: {si_sdr_values_noisy0}')
print(f'SI-SDR for microphone2 (dB) in Noise: {si_sdr_values_noisy1}')
print(f'SI-SDR for microphone3 (dB) in Noise: {si_sdr_values_noisy2}')
