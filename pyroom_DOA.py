import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
import math

# 参数设置
fs = 16000  # 采样率
n_mics = 3  # 麦克风数量
signal_length = 47520  # 信号长度
room_dim = [8, 9]  # 房间尺寸

# 创建一个矩形房间
room = pra.ShoeBox(room_dim, fs=fs, absorption=1.0, max_order=0)

# 添加麦克风阵列
mic_positions = np.array([[4,6,5,4.5,5.5,2,3,7,8],  # x 坐标
                          [6,6,6,6,6,6,6,6,6]])  # y 坐标
mic_array = pra.MicrophoneArray(mic_positions, fs)
room.add_microphone_array(mic_array)

# 添加信号源
source_position = [2, 7]  # 信号源位置
source_signal = np.random.randn(signal_length)  # 模拟信号源
room.add_source(source_position, signal=source_signal)

# 计算房间的RIR（Room Impulse Response）
room.compute_rir()

# 模拟麦克风接收信号（将信号与房间的RIR卷积）
received_signals = room.simulate(return_premix=True)
received_signals=np.squeeze(received_signals)
#print(received_signals.shape)
# 使用 SRP-PHAT 进行 DOA 估计
X = np.array(
    [
        pra.transform.stft.analysis(signal, 256, 256 // 2).T
        for signal in received_signals
    ]
)
print(X.shape)

doa = pra.doa.music.MUSIC(mic_array.R, fs, nfft=256, num_src=1)  # 初始化 MUSIC DOA 估计算法
doa.locate_sources(X)  # 定位信号源
print(doa.azimuth_recon / np.pi * 180.0)
print(math.atan2(5-2,6-7)*180/math.pi)

#estimated_doa=np.rad2deg(doa.azimuth_recon)
#beamformer=pra.Beamformer(mic_array.R,fs=fs,N=256)
#steering_vector=pra.beamforming.compute_steering_vector(mic_array,doa.azimuth_recon[0],fs)



'''def compute_covariance_matrix(received_signals):
    """
    计算接收信号的协方差矩阵。

    参数:
    - received_signals: numpy 数组，形状为 (M, N)，M 是麦克风数量，N 是信号长度

    返回:
    - R: numpy 数组，接收信号的协方差矩阵，形状为 (M, M)
    """
    R = np.dot(received_signals, received_signals.T) / received_signals.shape[1]
    return R

def compute_steering_vector(mic_positions, doa, frequency, speed_of_sound=343):
    """
    计算导向矢量（Steering Vector）。

    参数:
    - mic_positions: numpy 数组，麦克风阵列的位置，形状为 (M, 2)，M 是麦克风数量
    - doa: float，信号的 DOA（以弧度表示）
    - frequency: float，信号的频率（Hz）
    - speed_of_sound: float，声速，默认为 343 m/s

    返回:
    - steering_vector: numpy 数组，导向矢量，形状为 (M, 1)
    """
    wavelength = speed_of_sound / frequency
    k = 2 * np.pi / wavelength
    steering_vector = np.exp(-1j * k * (mic_positions[:, 0] * np.cos(doa) + mic_positions[:, 1] * np.sin(doa)))
    return steering_vector

def compute_mvdr_weights(R, steering_vector):
    """
    计算 MVDR 波束形成权重。

    参数:
    - R: numpy 数组，协方差矩阵，形状为 (M, M)
    - steering_vector: numpy 数组，导向矢量，形状为 (M, 1)

    返回:
    - weights: numpy 数组，MVDR 权重，形状为 (M, 1)
    """
    R_inv = np.linalg.inv(R)
    numerator = np.dot(R_inv, steering_vector)
    denominator = np.dot(steering_vector.conj().T, numerator)
    weights = numerator / denominator
    return weights

def apply_mvdr_beamforming(received_signals, weights):
    """
    应用 MVDR 波束形成。

    参数:
    - received_signals: numpy 数组，接收到的信号（形状为 (M, N)，M 是麦克风数量，N 是信号长度）
    - weights: numpy 数组，MVDR 权重（形状为 (M, 1)）

    返回:
    - beamformed_signal: numpy 数组，波束形成后的信号（形状为 (N,)）
    """
    beamformed_signal = np.dot(weights.conj().T, received_signals)
    return beamformed_signal.flatten()

# 计算协方差矩阵
R = compute_covariance_matrix(received_signals)

# 计算导向矢量
steering_vector = compute_steering_vector(mic_positions, doa, frequency)

# 计算 MVDR 权重
mvdr_weights = compute_mvdr_weights(R, steering_vector)

# 应用 MVDR 波束形成
beamformed_signal = apply_mvdr_beamforming(received_signals, mvdr_weights)

# 绘制波束形成后的结果
plt.figure()
plt.plot(beamformed_signal)
plt.title('MVDR Beamformed Signal')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.grid()
plt.show()'''



