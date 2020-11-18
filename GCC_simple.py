import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as k
import librosa
import tensorflow as tf
import numpy as np
import pyroomacoustics as pra
from sklearn.cluster import KMeans


def end_padding(signal, target_length):
    gap = target_length - len(signal)
    signal = np.pad(signal, (0, gap), mode="constant")
    return signal, gap


def gccphat(recv_signals0):
    recv_signals = tf.signal.frame(recv_signals0, frame_length=256, frame_step=32).numpy() * np.hanning(256)[np.newaxis,
                                                                                                             np.newaxis]
    recv_signals = np.fft.rfft(recv_signals)
    est_power = np.abs(recv_signals)
    frame_level_power = k.sum(tf.cast(est_power, tf.float32), axis=[0, 2]).numpy()
    angle = recv_signals / np.abs(recv_signals)
    time_shift = np.arange(-(2 * r) / 340, (2 * r) / 340, (2 * r) / 340 / 90)
    freq_arange = np.fft.rfftfreq(256, 1 / 8000)
    ft = time_shift[np.newaxis] * freq_arange[..., np.newaxis]
    gcc_term = np.exp(1j * np.pi * 2 * ft)
    gccphat = np.real(
        k.sum(angle[0][..., np.newaxis] * angle[1].conj()[..., np.newaxis] * gcc_term[np.newaxis], axis=1))
    gccphat = np.abs(gccphat)
    # gccphat /= np.max(gccphat, axis=-1, keepdims=True)
    gccphat = tf.nn.softmax(gccphat, axis=-1)* frame_level_power[:, np.newaxis]
    plt.imshow(gccphat)
    plt.show()
    plt.plot(tf.nn.softmax(np.mean(gccphat, axis=0)))
    plt.show()
    max_angle = np.argmax(np.mean(gccphat, axis=0))
    print(max_angle)
    return max_angle


def makeroom(source, room_sz, t60, normalize_macs):
    num_source = len(source)
    max_dist = min(room_sz[0], room_sz[1]) / 2.
    room_center = room_sz / 2.
    mac_hight = np.random.uniform(low=0.6, high=1)
    normalize_macs[2, :] += mac_hight
    normalize_macs[:2, :] += room_center[:2, np.newaxis]
    rectfied_macs = pra.MicrophoneArray(normalize_macs, 8000)
    e_absorption, max_order = pra.inverse_sabine(rt60=t60, room_dim=room_sz)
    room = pra.ShoeBox(room_sz, absorption=e_absorption, max_order=0, mics=rectfied_macs)
    spk_dist = np.random.uniform(low=max_dist/2., high=max_dist, size=[num_source])
    shift = np.random.uniform(0, np.pi*2)
    spk_angle = np.asarray([np.pi/5, np.pi/5+np.pi/3])
    print((spk_angle/np.pi/2*360)%180)
    spk_x = np.cos(spk_angle) * spk_dist + room_center[0]
    spk_y = np.sin(spk_angle) * spk_dist + room_center[1]
    spk_z = np.random.uniform(1.5, 2.0, size=num_source)
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
    data, gaps = makeroom(audios, room_sz=np.asarray([w, l, h]), t60=.5, normalize_macs=normalize_macs)

    recv_signals = data[:7]
    label_signals = data[7:]

    from scipy.io import wavfile
    # recv_signals = np.load("haha.npy").T/np.max(recv_signals)
    # recv_signals = librosa.resample(recv_signals, orig_sr=48000, target_sr=8000)
    gccphat(np.asarray([recv_signals[0], recv_signals[3]]))
    recv_signals = recv_signals / np.max(recv_signals, keepdims=False)
    wavfile.write("sameplerec.wav", rate=8000, data=recv_signals[-1])
    recv_signals0 = recv_signals.astype(np.float32)
    spectral = tf.signal.stft(recv_signals0, frame_length=512, frame_step=64, fft_length=512).numpy()
    angle = np.arctan2(np.real(spectral), np.imag(spectral))
    plt.imshow(angle[0] - angle[1])
    plt.show()
    plt.imshow((np.sin(angle[0] - angle[1])))
    plt.show()
    sinipd = np.sin(angle[0] - angle[1])
