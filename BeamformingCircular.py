import matplotlib.pyplot as plt
import librosa
import tensorflow as tf
import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
from sklearn.decomposition import NMF


def end_padding(signal, target_length):
    gap = target_length - len(signal)
    signal = np.pad(signal, (0, gap), mode="constant")
    return signal, gap


def makeroom(source, room_sz, t60, normalize_macs):
    num_source = len(source)
    max_dist = min(room_sz[0], room_sz[1]) / 2.
    room_center = room_sz / 2.
    mac_hight = np.random.uniform(low=0.6, high=1)
    normalize_macs[2, :] += mac_hight
    normalize_macs[:2, :] += room_center[:2, np.newaxis]
    rectfied_macs = pra.MicrophoneArray(normalize_macs, 8000)
    e_absorption, max_order = pra.inverse_sabine(rt60=t60, room_dim=room_sz)
    room = pra.ShoeBox(room_sz, absorption=e_absorption, max_order=max_order, mics=rectfied_macs)
    spk_dist = np.random.uniform(low=max_dist/2., high=max_dist, size=[num_source])
    spk_angle = np.asarray([0, np.pi/3])
    print((spk_angle/np.pi/2*360))
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


def mu_law_encode(audio, quantization_channels):
    mu = quantization_channels - 1
    safe_audio_abs = np.abs(np.clip(audio, a_min=-1, a_max=1))
    magnitude = np.log1p(mu * safe_audio_abs) / np.log1p(mu)
    signal = np.sign(audio) * magnitude
    return ((signal + 1) / 2 * mu + 0.5).astype(np.int16)


def mu_law_decode(output, quantization_channels):
    mu = quantization_channels - 1
    signal = 2 * (output / mu) - 1
    magnitude = (1 / mu) * ((1 + mu) ** np.abs(signal) - 1)
    return np.sign(signal) * magnitude


if __name__ == '__main__':
    fs = 8000
    r = 0.02
    fft_length = 512
    step = 128
    drop_out = .01
    drop_band_lower = 20
    drop_band_upper = 20
    """audios = []
    for i in range(2):
        audios.append(
            librosa.load("/root/PycharmProjects/pythonProject/A-U-net_V2.0/output/sample_clean{}.wav".format(i),
                         sr=8000)[0])
    w = 6
    l = 4
    h = 3
    r = .02
    normalize_macs = pra.beamforming.circular_2D_array(center=[0, 0],
                                                       M=6,
                                                       radius=r,
                                                       phi0=0)
    normalize_macs = np.concatenate([normalize_macs, np.zeros([1, 6])], axis=0)
    data, gaps = makeroom(audios, room_sz=np.asarray([w, l, h]), t60=.5, normalize_macs=normalize_macs)

    recv_signals = data[:6]
    label_signals = data[6:]"""
    recv_signals = np.load("/home/jax/Desktop/haha.npy").T.astype(np.float64)
    recv_signals = librosa.resample(recv_signals, orig_sr=45000, target_sr=8000)
    recv_signals -= np.mean(recv_signals)
    recv_signals /= np.max(np.abs(recv_signals))
    recv_signals = mu_law_decode(mu_law_encode(recv_signals, quantization_channels=512), quantization_channels=512)
    recv_signals = recv_signals.astype(np.float32)
    wavfile.write("sameplerec.wav", rate=fs, data=recv_signals[-2])
    spectral = tf.signal.stft(recv_signals, frame_length=fft_length, frame_step=step, fft_length=fft_length).numpy()
    spectral = spectral[..., drop_band_lower:-drop_band_upper]
    normed_spectral = np.exp(1j*np.angle(spectral))
    power_spectral = np.abs(spectral)
    counting, datarange = np.histogram(power_spectral.reshape([-1]), bins=1000)
    rank = -1
    counter = 0
    drop_out = (1 - drop_out) * power_spectral.size
    for rank, i in enumerate(counting):
        counter += i
        if counter > drop_out:
            break
    threhold = datarange[rank+1]
    power_spectral = np.clip(power_spectral, a_min=0, a_max=threhold)
    spectral = power_spectral * normed_spectral

    chs, time, freq = spectral.shape
    rxx = np.einsum("itf, jtf->tfij", spectral, spectral.conj())
    rxx_amp = np.abs(rxx)
    rxx_phase = np.angle(rxx)
    rxx_amp = tf.nn.sigmoid(rxx_amp).numpy()
    rxx = rxx_amp * np.exp(1j*rxx_phase)
    rxx = np.sum(rxx, axis=0)
    # weight = np.sqrt(np.mean(np.mean(power_spectral[..., 15:50]**2, axis=0), axis=-1))
    # weight /= np.sum(weight)

    # rxx = np.sum(rxx*weight[:, np.newaxis, np.newaxis, np.newaxis], axis=0)
    rxxinv = np.asarray([np.linalg.pinv(i) for i in rxx])
    # presision_angle = np.arange(0, 360, 1) / 180 * np.pi
    presision_angle = np.asarray([34, 134]) / 180 * np.pi
    mac_top_angle = np.arange(0, 360, 60) / 180 * np.pi
    time_delay = - r * np.cos(presision_angle[np.newaxis] - mac_top_angle[:, np.newaxis]) / 340
    freqrange = np.fft.rfftfreq(fft_length, 1 / fs)[drop_band_lower:-drop_band_upper]
    t_f = time_delay[..., np.newaxis] * freqrange[np.newaxis, np.newaxis]
    steermat = np.exp(- 1j * np.pi * 2 * t_f)
    # steermat = np.concatenate([steermat, np.ones([1, 360, 129])], axis=0)
    upper = np.einsum("fij, jaf->iaf", rxxinv, steermat)
    deameter = np.einsum("iaf, iaf->af", steermat.conj(), upper)[np.newaxis]
    wmvdr = upper / deameter
    # plt.plot(np.sum(tf.nn.softmax(np.abs(1/deameter)[0]*10, axis=0), axis=-1))
    # plt.show()
    fas = np.einsum("ctf, caf->atf", spectral, wmvdr.conj())
    fas = np.pad(fas, ((0, 0), (0, 0), (drop_band_lower, drop_band_upper)))
    fas = tf.signal.inverse_stft(fas, frame_length=fft_length, frame_step=step, fft_length=fft_length).numpy()
    power_fas = np.abs(fas)
    for i in range(2):
        plt.plot(fas[i])
        plt.show()
        wavfile.write("{}.wav".format(i), rate=fs, data=fas[i])
