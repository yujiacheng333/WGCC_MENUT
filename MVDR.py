import matplotlib.pyplot as plt
import librosa
import tensorflow as tf
import numpy as np
import pyroomacoustics as pra
from sklearn.cluster import KMeans


def end_padding(signal, target_length):
    gap = target_length - len(signal)
    signal = np.pad(signal, (0, gap), mode="constant")
    return signal, gap


def makeroom(source, room_sz, t60, normalize_macs):
    num_source = len(source)
    max_dist = min(room_sz[0], room_sz[1]) / 2.
    room_center = room_sz / 2.
    # mac_hight = np.random.uniform(low=0.6, high=1)
    mac_hight = 1.76
    normalize_macs[2, :] += mac_hight
    normalize_macs[:2, :] += room_center[:2, np.newaxis]
    rectfied_macs = pra.MicrophoneArray(normalize_macs, 8000)
    e_absorption, max_order = pra.inverse_sabine(rt60=t60, room_dim=room_sz)
    room = pra.ShoeBox(room_sz, absorption=e_absorption, max_order=max_order, mics=rectfied_macs)
    # spk_dist = np.random.uniform(low=max_dist/2., high=max_dist, size=[num_source])
    spk_dist = np.asarray([max_dist//2, max_dist//2])
    shift = np.random.uniform(np.pi/4*1, np.pi/4*3)
    spk_angle = np.asarray([np.pi/3, np.pi/5*8])
    print((spk_angle/np.pi/2*360))

    spk_x = np.cos(spk_angle) * spk_dist + room_center[0]
    spk_y = np.sin(spk_angle) * spk_dist + room_center[1]
    # spk_z = np.random.uniform(1.5, 2.0, size=num_source)
    spk_z = np.asarray([1.76, 1.76])
    for i in range(num_source):
        room.add_source(position=[spk_x[i], spk_y[i], spk_z[i]], signal=source[i])
    room.compute_rir()
    room.simulate()
    recv_signals = room.mic_array.signals
    recv_signals = recv_signals.astype(np.float32)

    """
    label_signal
    """
    label_signals = []
    gaps = []
    room_center[-1] = mac_hight
    room_center = room_center[:, np.newaxis]
    room_center_mac = pra.MicrophoneArray(room_center, 8000)
    for i in range(num_source):
        room = pra.ShoeBox(room_sz, absorption=1., max_order=0, mics=room_center_mac)
        room.add_source(position=[spk_x[i], spk_y[i], spk_z[i]], signal=source[i])
        room.compute_rir()
        room.simulate()
        pad_signal, gap = end_padding(room.mic_array.signals[0], target_length=recv_signals.shape[-1])
        label_signals.append(pad_signal)
        gaps.append(gap)
    one_step_data= tf.concat([recv_signals, tf.cast(label_signals, tf.float32)], axis=0).numpy()
    return one_step_data, np.max(gaps)


if __name__ == '__main__':
    audios = []
    for i in range(2):
        audios.append(librosa.load("A-U-net_V2.0/output/sample_clean{}.wav".format(i), sr=8000)[0])
    w = 6
    l = 4
    h = 3
    r = .04
    normalize_macs = pra.beamforming.circular_2D_array(center=[0, 0],
                                                       M=6,
                                                       radius=r,
                                                       phi0=0)
    center_mac = np.asarray([[0], [0]])
    normalize_macs = np.concatenate([normalize_macs, center_mac], axis=1)
    normalize_macs = np.concatenate([normalize_macs, np.zeros([1, 7])], axis=0)
    data, gaps = makeroom(audios, room_sz=np.asarray([w, l, h]), t60=.3, normalize_macs=normalize_macs)

    recv_signals = data[:7]
    label_signals = data[7:]

    spectral = tf.signal.stft(recv_signals, frame_length=256, frame_step=64)
    ref_pow = np.sum(np.abs(spectral[-1]), axis=1)
    spectral = spectral[:6]
    spectral = tf.transpose(spectral, [1, 0, 2]).numpy()  # T, C, F
    rxx = np.einsum("...if, ...jf->...ijf", spectral, spectral.conj()) # T C C F
    time, chs, _, freq = rxx.shape
    rxx = np.transpose(rxx, [0, 3, 1, 2]).reshape([1, time, freq, chs*chs])
    ref_pow = tf.nn.softmax(ref_pow/np.max(ref_pow)*5).numpy()
    rxx = np.repeat(np.sum(rxx*ref_pow[np.newaxis, :, np.newaxis, np.newaxis], axis=1, keepdims=True),
                    repeats=time, axis=1)
    rxx_real = tf.keras.layers.AveragePooling2D(pool_size=(1, 64), padding="same", strides=1)(np.real(rxx)).numpy()
    rxx_image = tf.keras.layers.AveragePooling2D(pool_size=(1, 64), padding="same", strides=1)(np.imag(rxx)).numpy()
    rxx = rxx_real + 1j*rxx_image
    rxx = rxx.reshape([time*freq, chs, chs])
    rxxinv = np.asarray([np.linalg.inv(i) for i in rxx])
    rxx = rxx.reshape([time, freq, chs, chs])
    rxxinv = rxxinv.reshape([time, freq, chs, chs])
    precision_angles = np.arange(0, 360, 1)/180 * np.pi
    mac_angles = np.arange(6) * np.pi/3  # 6
    freq_band = np.arange(129)/128 * 4000 # 129
    time_delay = - np.cos(mac_angles[:, np.newaxis] - precision_angles[np.newaxis])*r/340
    tfdot = time_delay[..., np.newaxis] * freq_band[np.newaxis, np.newaxis]
    steermat = np.exp(-1j*np.pi*2*tfdot)  # [mac, theta, freq]
    # steermat = np.concatenate([steermat, np.ones([1, 360, 129])], axis=0)
    upper = np.einsum("tfij, jaf->tiaf", rxxinv, steermat)
    deameter = np.einsum("iaf, tiaf->taf", steermat.conj(), upper)[:, np.newaxis]
    wmvdr = upper / (deameter + 1e-6)
    # wmvdr :409, 6, 360, 129  spectral 409, 6, 129 -> 409, 6, 360, 129
    fas = np.sum(wmvdr * spectral[..., np.newaxis, :], axis=1).transpose([1, 0, 2])
    fas = tf.cast(fas, tf.complex64)
    fas = tf.signal.inverse_stft(fas, frame_step=64, frame_length=256).numpy()
    power = np.sum(np.abs(fas), axis=-1)
    plt.plot(np.sum(np.abs(fas), axis=-1))
    plt.show()
    from scipy.io import wavfile
    plt.plot(np.clip(fas[60].astype(np.float32), -100, 100))
    plt.show()
    plt.plot(fas[288])
    plt.show()
    wavfile.write("0.wav", rate=8000, data=fas[60])
    wavfile.write("1.wav", rate=8000, data=fas[288].astype(np.float32))
