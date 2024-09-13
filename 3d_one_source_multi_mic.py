import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting tools
import math
import torch
import os
import json
import random
from pathlib import Path
import logging
import pandas as pd
import scaper
import torchaudio
import torchaudio.transforms as AT
from torch import tensor
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from random import randrange
from multiprocessing import Pool
import warnings
import time
from scipy.signal import stft, istft, resample
warnings.filterwarnings("ignore")

np.random.seed(3407)
#print(torch.cuda.get_device_name(0))
fs = 44100  # Sampling frequency
room_dim = [6, 4, 3]  # 3D room dimensions [length, width, height]

input_dir="C:/Users/yifengw3/Desktop/yifeng_project1/Waveformer/FSDSoundScapes"
dset='train'
fg_dir = os.path.join(input_dir, 'FSDKaggle2018/%s' % dset)
bg_dir = os.path.join(input_dir,'TAU-acoustic-sounds','TAU-urban-acoustic-scenes-2019-development')
samples = sorted(list(Path(os.path.join(input_dir, 'jams', dset)).glob('[0-9]*')))
jamsfile = os.path.join(samples[0], 'mixture.jams')
_, jams, _, _ = scaper.generate_from_jams(jamsfile, fg_path=fg_dir, bg_path=bg_dir)
_sr = jams['annotations'][0]['sandbox']['scaper']['sr']
#print(_sr)
#print(len(samples))
'''def compute_azimuth_elevation(pointA, pointB):
    vector = np.array([pointB[0] - pointA[0], pointB[1] - pointA[1], pointB[2] - pointA[2]])
    azimuth = np.arctan2(vector[1], vector[0])  
    azimuth_deg = np.degrees(azimuth)  
    r = np.linalg.norm(vector)  
    elevation = np.arcsin(vector[2] / r) 
    elevation_deg = np.degrees(elevation)  

    return azimuth_deg, elevation_deg'''

############### SNR metrics calculation
def calculate_si_snr(target, estimate):
    target = np.asarray(target).flatten()
    estimate = np.asarray(estimate).flatten()
    target_energy = np.sum(target ** 2)
    scale = np.dot(estimate, target) / target_energy
    s_target = scale * target
    e_noise = estimate - s_target
    si_snr = 10 * np.log10(np.sum(s_target ** 2) / np.sum(e_noise ** 2))
    return si_snr

def calculate_psnr(target, estimate):
    mse=np.mean((target-estimate)**2)
    max_value=np.max(target)
    psnr=10*np.log10(max_value**2/mse)
    return psnr

def calculate_snr(signal, noise):
    power_signal = np.mean(signal ** 2)
    power_noise = np.mean(noise ** 2)
    snr = 10 * np.log10(power_signal / power_noise)
    return snr

def calculate_sdr(original, estimated):
    error_signal = original - estimated
    sdr = 10 * np.log10(np.sum(original ** 2) / np.sum(error_signal ** 2))
    return sdr

def calculate_si_sdr(reference, estimated):
    reference = np.asarray(reference).flatten()
    estimated = np.asarray(estimated).flatten()
    alpha = np.dot(estimated, reference) / np.dot(reference, reference)
    projection = alpha * reference
    noise = estimated - projection
    si_sdr = 10 * np.log10(np.sum(projection ** 2) / np.sum(noise ** 2))
    return si_sdr

############# Beamforming for MVDR
def steering_vector(mic_positions, azimuth,elevation, sound_speed=343.0):
    azimuth, elevation = np.deg2rad(azimuth),np.deg2rad(elevation)
    direction = np.array([
        np.cos(azimuth) * np.cos(elevation),
        np.sin(azimuth) * np.cos(elevation),
        np.sin(elevation)
    ])
    #print(direction.shape,mic_positions.T.shape)
    return np.exp(-1j * (mic_positions.T @ direction))

def mvdr_weights_broadband(R_n, a_theta, epsilon=0.005):
    #print(a_theta.shape)
    n_mics, _, n_freq_bins = R_n.shape
    weights_mvdr = np.zeros((n_mics, n_freq_bins), dtype=complex)
    for f in range(n_freq_bins):
        R_n_inv = np.linalg.inv(R_n[:, :, f] + epsilon * np.eye(n_mics))  
        numerator = R_n_inv @ a_theta
        denominator = a_theta.conj().T @ R_n_inv @ a_theta
        weights_mvdr[:, f] = numerator / denominator  
    return weights_mvdr

def MVDR(mic_signals,n_fft,hop_length,n_mics,mic_positions,doa_azimuth, doa_elevation):
    f, t, stft_signals = zip(*[stft(mic_signals[m, :], fs=fs, nperseg=n_fft, noverlap=hop_length) for m in range(n_mics)])
    stft_signals = np.array(stft_signals)
    n_freq_bins, n_frames = stft_signals[0].shape
    R_n = np.zeros((n_mics, n_mics, n_freq_bins), dtype=complex)
    for f in range(n_freq_bins):
        X_f = stft_signals[:, f, :]  
        R_n[:, :, f] = np.dot(X_f, X_f.conj().T) / X_f.shape[1]  
    a_theta = steering_vector(mic_positions, doa_azimuth, doa_elevation)
    weights_mvdr = mvdr_weights_broadband(R_n, a_theta)
    output_stft = np.zeros_like(stft_signals[0], dtype=complex)
    for f in range(n_freq_bins):
        for t in range(n_frames):
            X_f_t = stft_signals[:, f, t]
            output_stft[f, t] = weights_mvdr[:, f].conj().T @ X_f_t
    _, output_signal = istft(output_stft, fs=fs, nperseg=n_fft, noverlap=hop_length)
    return output_signal

########## Beamforming for DAS
def delay_and_sum_beamforming(mic_signals, mic_positions, fs, azimuth,elevation, sound_speed=343.0):
    n_mics, n_samples = mic_signals.shape
    azimuth, elevation = np.deg2rad(azimuth),np.deg2rad(elevation)
    direction_vector = np.array([
        np.cos(azimuth) * np.cos(elevation),
        np.sin(azimuth) * np.cos(elevation),
        np.sin(elevation)
    ])
    delays = np.dot(mic_positions.T, direction_vector) / sound_speed
    sample_delays = np.round(delays * fs).astype(int)
    aligned_signals = np.zeros((n_mics, n_samples + np.max(np.abs(sample_delays))))
    for i in range(n_mics):
        delay = sample_delays[i]
        if delay > 0:
            aligned_signals[i, delay:delay + n_samples] = mic_signals[i]
        else:
            aligned_signals[i, :n_samples + delay] = mic_signals[i, -delay:]
    output_signal = np.sum(aligned_signals, axis=0)
    output_signal = output_signal[:n_samples]
    return output_signal

########## DOA true value calculation
def compute_azimuth_elevation(direction_vector):
    azimuth = np.arctan2(direction_vector[1], direction_vector[0])  # atan2(dy, dx)
    azimuth_deg = np.rad2deg(azimuth)
    elevation = np.arctan2(direction_vector[2], np.sqrt(direction_vector[0]**2 + direction_vector[1]**2))
    elevation_deg = np.rad2deg(elevation)
    return azimuth_deg,elevation_deg

########## Spherical Microphone
def spherical_3D_array(center, radius, num_mics):
    indices = np.arange(0, num_mics, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / num_mics)  
    theta = np.pi * (1 + 5**0.5) * indices  
    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(phi)
    mic_positions = np.vstack((x, y, z))  
    return mic_positions


'''mic_positions = np.array([[3-0.1238/2, 3+0.1238/2],  # x-coordinates
                            [2, 2],          # y-coordinates
                            [1.5, 1.5+0.0076/2]])  # z-coordinates '''

# Define 3D microphone array positions (x, y, z)
'''mic_positions = np.array([[3-0.1238/2, 3+0.1238/2, 3+0.1238/4],  # x-coordinates
                        [2, 2, 2-0.0586/4],          # y-coordinates
                        [1.5, 1.5+0.0076/2, 1.5-0.0076/2]])  # z-coordinates '''

'''mic_positions = np.array([[3-0.1238/2, 3+0.1238/2, 3+0.1238/4,3-0.1238/2,3-0.1238/2],  # x-coordinates
                        [2, 2, 2-0.0586/4,2-0.1,2+0.1],          # y-coordinates
                        [1.5, 1.5+0.0076/2, 1.5-0.0076/2,1.5+0.85,1.5+0.85]])  # z-coordinates '''

'''mic_positions = np.array([[3-0.025*7,3-0.025*5,3-0.025*3,3-0.025*1,3+0.025*1,3+0.025*3,3+0.025*5,3+0.025*7],  # x-coordinates
                        [2, 2, 2,2,2,2,2,2],          # y-coordinates
                        [1.5, 1.5,1.5, 1.5,1.5, 1.5,1.5, 1.5]])  # z-coordinates'''

'''mic_positions = np.array([[3-0.025*3,3-0.025*1,3+0.025*1,3+0.025*3],  # x-coordinates
                        [2, 2, 2,2],          # y-coordinates
                        [1.5, 1.5,1.5, 1.5]])  # z-coordinates'''
'''mic_positions_total=[np.array([[3-0.1238/2, 3+0.1238/2, 3+0.1238/4], [2, 2, 2-0.0586/4], [1.5, 1.5+0.0076/2, 1.5-0.0076/2]]),
                    np.array([[3-0.1238/2, 3+0.1238/2, 3+0.1238/4,3-0.1238/2,3-0.1238/2], [2, 2, 2-0.0586/4,2-0.1,2+0.1], [1.5, 1.5+0.0076/2, 1.5-0.0076/2,1.5+0.85,1.5+0.85]]),
                    np.array([[3-0.025*1,3+0.025*1], [2, 2], [1.5, 1.5]]),
                    np.array([[3-0.025*3,3-0.025*1,3+0.025*1,3+0.025*3], [2, 2, 2,2], [1.5, 1.5,1.5, 1.5]]),
                    np.array([[3-0.025*5,3-0.025*3,3-0.025*1,3+0.025*1,3+0.025*3,3+0.025*5], [2, 2, 2,2,2,2], [1.5, 1.5,1.5, 1.5,1.5,1.5]]),
                    np.array([[3-0.025*7,3-0.025*5,3-0.025*3,3-0.025*1,3+0.025*1,3+0.025*3,3+0.025*5,3+0.025*7], [2, 2, 2,2,2,2,2,2], [1.5, 1.5,1.5, 1.5,1.5, 1.5,1.5, 1.5]])
                    ]'''
#create_room=
mic_positions_total=[np.array([[3-0.1238/2, 3+0.1238/2, 3+0.1238/4], [2, 2, 2-0.0586/4], [1.5, 1.5+0.0076/2, 1.5-0.0076/2]]),
                    np.array([[3-0.1238/2, 3+0.1238/2, 3+0.1238/4,3-0.1238/2,3-0.1238/2], [2, 2, 2-0.0586/4,2-0.1,2+0.1], [1.5, 1.5+0.0076/2, 1.5-0.0076/2,1.5+0.85,1.5+0.85]]),
                    spherical_3D_array([3,2,1.5], 1, 4),
                    spherical_3D_array([3,2,1.5], 1, 8),
                    spherical_3D_array([3,2,1.5], 1, 12),
                    np.hstack((np.array([[3-0.1238/2, 3+0.1238/2, 3+0.1238/4,3-0.1238/2,3-0.1238/2], [2, 2, 2-0.0586/4,2-0.1,2+0.1], [1.5, 1.5+0.0076/2, 1.5-0.0076/2,1.5+0.85,1.5+0.85]]),spherical_3D_array([3,2,1.5], 1, 4)))
                    ]
'''mic_positions_total=[np.array([[3-0.1238/2, 3+0.1238/2, 3+0.1238/4], [2, 2, 2-0.0586/4], [1.5, 1.5+0.0076/2, 1.5-0.0076/2]]),
                    np.array([[3-0.1238/2, 3+0.1238/2, 3+0.1238/4,3-0.1238/2,3-0.1238/2], [2, 2, 2-0.0586/4,2-0.1,2+0.1], [1.5, 1.5+0.0076/2, 1.5-0.0076/2,1.5+0.85,1.5+0.85]])]'''
#print(mic_positions_total[2])
source_positions = np.column_stack((
    np.random.uniform(0, 6, size=50),  
    np.random.uniform(0, 4, size=50),  
    np.random.uniform(0, 3, size=50)   
)).tolist()
#print(mic_positions_total[1].shape)
# Define a 3D source position
#source_position = [[5, 1, 0.5]]  # [x, y, z] coordinates for the source
def process_mic_positions(mic_positions):
    azimuth_diff_list=[]
    elevation_diff_list=[]
    sisnr_list1=[]
    psnr_list1=[]
    sdr_list1=[]
    sisdr_list1=[]
    sisnr_list2=[]
    psnr_list2=[]
    sdr_list2=[]
    sisdr_list2=[]
    #sisnr_list3=[]
    #psnr_list3=[]
    #sdr_list3=[]
    #sisdr_list3=[]
    snr_total=[sisnr_list1,psnr_list1,sdr_list1,sisdr_list1,sisnr_list2,psnr_list2,sdr_list2,sisdr_list2]
    for source_position in source_positions:
        for id in range(5):
            #print(id)
            sample_path = samples[id]       #config
            jamsfile = os.path.join(sample_path, 'mixture.jams')
            try:
                mixture, jams, ann_list, event_audio_list = scaper.generate_from_jams(jamsfile, fg_path=fg_dir, bg_path=bg_dir)
            except:
                continue
            isolated_events = {}
            for e, a in zip(ann_list, event_audio_list[1:]):
                # 0th event is background
                isolated_events[e[2]] = a
            gt_events = list(pd.read_csv(
                os.path.join(sample_path, 'gt_events.csv'), sep='\t')['label'])
            gt = torch.zeros_like(torch.from_numpy(event_audio_list[1]).permute(1, 0))
            gt=gt+torch.from_numpy(isolated_events[gt_events[0]]).permute(1,0)
            mixture = np.float64(mixture)
            #mixture=np.float64(gt)
            #print(mixture.shape)


            # Create a 3D room
            room = pra.ShoeBox(room_dim, fs=fs, absorption=0.2, max_order=10)
            #room=pra.AnechoicRoom(fs=fs)

            # Create the microphone array
            mic_array = pra.MicrophoneArray(mic_positions, fs)
            #print(mic_positions.shape)
            room.add_microphone_array(mic_array)

            source_signal = np.squeeze(mixture)
            room.add_source(source_position, signal=source_signal)

            # Compute room impulse responses (RIRs)
            room.compute_rir()

            # Simulate received signals at the microphone array
            #room.simulate(return_premix=True)
            room.simulate()
            received_signals=room.mic_array.signals
            #print(received_signals.shape)
            received_signals = np.squeeze(received_signals)
            #print(received_signals.shape)
            # Perform STFT on the received signals
            X = np.array(
                [
                    pra.transform.stft.analysis(signal, 512, 512 // 2).T
                    for signal in received_signals
                ]
            )

            #print(f"STFT Shape: {X.shape}")  # Output the shape of the STFT

            # Initialize MUSIC DOA estimation in 3D
            doa = pra.doa.music.MUSIC(mic_array.R, fs, nfft=512, num_src=1, dim=3,mode="near")
            #doa = pra.doa.srp.SRP(mic_array.R, fs, nfft=512, num_src=1, dim=3)
            #doa=pra.doa.cssm.CSSM(mic_array.R, fs, nfft=512, num_src=1, dim=3)
            #doa=pra.doa.waves.WAVES(mic_array.R, fs, nfft=512, num_src=1, dim=3)
            doa.locate_sources(X)  # Locate the sources using the MUSIC algorithm

            #doa.plot_individual_spectrum()
            # Output estimated DOA results
            estimated_doa = np.rad2deg(doa.azimuth_recon)  # Convert to degrees
            estimated_ele = 90-np.rad2deg(doa.colatitude_recon)
            #print(f"Estimated DOA (azimuth, elevation): {estimated_doa, estimated_ele} degrees")
            true_azi, true_ele = compute_azimuth_elevation(source_position-np.mean(mic_positions,axis=1))
            #print(f"True DOA (azimuth, elevation): {true_azi, true_ele} degrees")
            #print("dif for azimuth and elevation:",true_azi-estimated_doa.tolist(),true_ele-estimated_ele.tolist())
            azi_diff=abs(true_azi-estimated_doa.tolist())
            if azi_diff>180:
                azi_diff=360-azi_diff
            ele_diff=abs(true_ele-estimated_ele.tolist())
            if ele_diff>180:
                ele_diff=360-ele_diff
            azimuth_diff_list.append(azi_diff)
            elevation_diff_list.append(ele_diff)
            #print(mic_positions.shape[1])
            # Beamforming
            #print(estimated_doa,true_azi)
            #received_signals_resampled=resample(np.mean(received_signals,axis=0), len(source_signal))
            output_MVDR=MVDR(received_signals,n_fft=1024,hop_length=256,n_mics=mic_positions.shape[1],mic_positions=mic_positions,doa_azimuth=estimated_doa[0], doa_elevation=estimated_ele[0])
            #output_MVDR=delay_and_sum_beamforming(received_signals, mic_positions, fs, estimated_doa[0],estimated_ele[0], sound_speed=343.0)
            output_signal_resampled = resample(output_MVDR, len(source_signal))
            output_MVDR_perfect=MVDR(received_signals,n_fft=1024,hop_length=256,n_mics=mic_positions.shape[1],mic_positions=mic_positions,doa_azimuth=true_azi, doa_elevation=true_ele)
            #output_MVDR_perfect=delay_and_sum_beamforming(received_signals, mic_positions, fs, true_azi,true_ele, sound_speed=343.0)
            output_signal_resampled_perfect = resample(output_MVDR_perfect, len(source_signal))
            signal_total=[output_signal_resampled,output_signal_resampled_perfect]
            for i in range(len(signal_total)):
                '''if i==0:
                    for t in range(len(signal_total[0])):
                        received_signals_resampled=resample(signal_total[0][t], len(source_signal))
                        sisnr=calculate_si_snr(source_signal,received_signals_resampled)
                        psnr=calculate_psnr(source_signal,received_signals_resampled)
                        sdr=calculate_sdr(source_signal,received_signals_resampled)
                        sisdr=calculate_si_sdr(source_signal,received_signals_resampled)
                        snr_total[i*4+0].append(sisnr)
                        snr_total[i*4+1].append(psnr)
                        snr_total[i*4+2].append(sdr)
                        snr_total[i*4+3].append(sisdr)'''      #No need to calculate the original signal before beamforming
                #else:
                #sisnr=calculate_si_snr(source_signal,signal_total[i])
                si_snr = ScaleInvariantSignalNoiseRatio()
                sisnr=si_snr(torch.from_numpy(signal_total[i]), torch.from_numpy(source_signal))
                #print(sisnr)
                psnr=calculate_psnr(source_signal,signal_total[i])
                sdr=calculate_sdr(source_signal,signal_total[i])
                #sisdr=calculate_si_sdr(source_signal,signal_total[i])
                si_sdr = ScaleInvariantSignalDistortionRatio()
                sisdr=si_sdr(torch.from_numpy(signal_total[i]), torch.from_numpy(source_signal))
                #print(sisdr)
                snr_total[i*4+0].append(sisnr)
                snr_total[i*4+1].append(psnr)
                snr_total[i*4+2].append(sdr)
                snr_total[i*4+3].append(sisdr)
            #print(output_MVDR.shape,received_signals.shape)
    #print("azi",np.mean(np.array(azimuth_diff_list)),"ele",np.mean(np.array(elevation_diff_list)))
    return np.mean(np.array(azimuth_diff_list)),np.mean(np.array(elevation_diff_list)),snr_total

if __name__ == '__main__':
    start=time.time()
    f=open("mixture_0.2_10_50points.txt","a")
    for i in mic_positions_total:
        azidiff,elediff,snr_total=process_mic_positions(i)
        #with Pool(processes=os.cpu_count()) as pool:
        #results=list(pool.map(process_mic_positions, mic_positions_total))
        f.write("azidiff:")
        f.write(str(azidiff))
        f.write(" elediff:")
        f.write(str(elediff))
        #print(azidiff,elediff)
        for t,snr in enumerate(snr_total):
            if t%4==0:
                f.write('\n')
                f.write("sisnr:")
                f.write(str(np.around(np.mean(np.array(snr)),4)))
                #print("sisnr",np.mean(np.array(snr)))
            elif t%4==1:
                f.write(" psnr:")
                f.write(str(np.around(np.mean(np.array(snr)),4)))
                #print("psnr",np.mean(np.array(snr)))
            elif t%4==2:
                f.write(" sdr:")
                f.write(str(np.around(np.mean(np.array(snr)),4)))
                #print("sdr",np.mean(np.array(snr)))
            elif t%4==3:
                f.write(" sisdr:")
                f.write(str(np.around(np.mean(np.array(snr)),4)))
                #print("sisdr",np.mean(np.array(snr)))
        f.write("\n")
        f.write("\n")
    end=time.time()
    f.close()
    print(end-start)




'''room = pra.ShoeBox(room_dim, fs=fs, absorption=1.0, max_order=1)
#room=pra.AnechoicRoom(fs=fs)

# Create the microphone array
mic_positions=mic_positions_total[5]
mic_array = pra.MicrophoneArray(mic_positions, fs)
#room.add_microphone_array(mic_array)

# Generate a random source signal
source_signal = np.zeros(44100)
#for source_position in source_positions:
#room.add_source(source_positions[0], signal=source_signal)
# Plot the room and microphone and source positions in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the room
room.plot(img_order=0, ax=ax)

# Plot microphone positions
ax.scatter(mic_positions[0, :], mic_positions[1, :], mic_positions[2, :], c='b', marker='o', label='Microphones')

# Plot source position
source_position_plot = np.array([source_positions]).T
ax.scatter(source_position_plot[0, :], source_position_plot[1, :], source_position_plot[2, :], c='r', marker='x', label='Signal Source')

# Set labels
ax.set_xlim([0, room_dim[0]])
ax.set_ylim([0, room_dim[1]])
ax.set_zlim([0, room_dim[2]])
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
plt.legend()
plt.title('3D Room Layout with Microphone Array and Source')
plt.show()
'''