import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as k
import librosa
import tensorflow as tf
import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
from sklearn.decomposition import NMF
from pyroomacoustics.bss.common import projection_back


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
    spk_dist = 1
    shift = np.random.uniform(np.pi / 8 * 1, np.pi / 8 * 3)
    spk_angle = np.asarray([np.pi / 8 * 1, np.pi / 8 * 7])
    print((spk_angle / np.pi / 2 * 360))
    spk_x = np.cos(spk_angle) * spk_dist + room_center[0]
    spk_y = np.sin(spk_angle) * spk_dist + room_center[1]
    spk_z = np.random.uniform(1.5, 2.0, size=num_source)
    for i in range(num_source):
        room.add_source(position=[spk_x[i], spk_y[i], spk_z[i]], signal=source[i])
    room.compute_rir()
    room.simulate()
    recv_signals = room.mic_array.signals
    recv_signals = recv_signals.astype(np.float32)
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
    one_step_data = tf.concat([recv_signals, tf.cast(label_signals, tf.float32)], axis=0).numpy()
    return one_step_data, np.max(gaps)


if __name__ == '__main__':
    ref_index = 0
    fs = 8000
    r = 0.02
    fft_length = 512
    step = 128
    drop_band_lower = 10
    drop_band_upper = 10
    method = "GDS_NMF"
    prefix = "/home/jax/Desktop/datas2/DATA4"
    with open(prefix + "/DATA.TXT", "rb") as f:
        recv_signals = f.read()
    recv_signals = tf.io.decode_raw(recv_signals, tf.int32)[2*8000:-8000*2]
    recv_signals = tf.reshape(recv_signals, [-1, 8]).numpy().T[:-1].astype(np.float)
    recv_signals -= np.mean(recv_signals)
    recv_signals /= np.max(np.abs(recv_signals))
    spectral = tf.signal.stft(recv_signals, frame_length=fft_length, frame_step=step, fft_length=fft_length).numpy()[..., drop_band_lower:-drop_band_upper]
    power = np.abs(spectral)
    normed_complex_spectral = np.exp(1j*np.angle(spectral))
    # steer matrix
    presision_angle = np.arange(0, 360, 1) / 180 * np.pi
    mac_top_angle = np.arange(0, 360, 60) / 180 * np.pi
    time_delay = - r * np.cos(presision_angle[np.newaxis] - mac_top_angle[:, np.newaxis]) / 340
    freqrange = np.fft.rfftfreq(fft_length, 1 / fs)[drop_band_lower:-drop_band_upper]
    t_f = time_delay[..., np.newaxis] * freqrange[np.newaxis, np.newaxis]
    steermat = np.exp(- 1j * np.pi * 2 * t_f)
    steermat = np.concatenate([np.ones([1, steermat.shape[1], steermat.shape[2]]), steermat], axis=0)
    gcc_map = tf.nn.softmax(1000*np.abs(np.einsum("caf, ctf->atf", steermat.conj(), normed_complex_spectral)), axis=0).numpy()
    spatial_power = gcc_map * power[ref_index][np.newaxis]
    spatial_power = np.sum(np.sum(spatial_power, axis=-1), axis=-1)
    spatial_power /= np.max(spatial_power)
    spatial_power[spatial_power < np.max(spatial_power)*2e-1] = 0

    pooling_window = 9
    left_padding = spatial_power[-(pooling_window // 2):]
    right_padding = spatial_power[:(pooling_window // 2)]
    spatial_power = np.concatenate([left_padding, spatial_power, right_padding], axis=0)
    temp_power = []
    for i in range(len(presision_angle)):
        selction = spatial_power[i:i + pooling_window]
        temp_power.append(np.mean(selction))
    plt.polar(presision_angle, temp_power)
    plt.show()
    spatial_power = temp_power
    search_window = 55
    outputs = []

    left_padding = spatial_power[-(search_window // 2):]
    right_padding = spatial_power[:(search_window // 2)]
    spatial_power = np.concatenate([left_padding, spatial_power, right_padding], axis=0)
    for i in range(len(presision_angle)):
        selction = spatial_power[i:i + search_window]
        if np.sum(selction[search_window // 2 + 1] > selction) == search_window-1:
            outputs.append(i)
    outputs = [0, 90]
    selected_steermat = steermat[:, outputs]
    if method is "GDS_NMF":
        nmf = NMF(n_components=40, alpha=0.1, beta_loss="kullback-leibler", max_iter=600, tol=1e-2, solver="mu")
        w = nmf.fit_transform(power[ref_index].T)
        h = nmf.components_
        weight = w / np.sqrt(np.sum(w**2, axis=0))
        masked_angle = np.einsum("ctf, fd->ctfd", normed_complex_spectral, weight)
        proj = np.abs(np.einsum("ctfd, caf->dtfa", masked_angle, selected_steermat.conj()))
        gcc_nmf = tf.one_hot(np.argmax(proj, axis=-1), depth=len(outputs)).numpy()
        recon_spectral = np.einsum("dt, dtfa->dtfa", h, gcc_nmf)
        recon_spectral = np.einsum("fd, dtfa->atf", w, recon_spectral)
    elif method is "GDS_NMF_org":
        nmf = NMF(n_components=fft_length, alpha=0.1, beta_loss="kullback-leibler", max_iter=600, tol=1e-2, solver="mu")
        w = nmf.fit_transform(power[ref_index].T)
        h = nmf.components_
        weight = w / np.sqrt(np.sum(w ** 2, axis=0))
        masked_angle = np.einsum("ctf, fd->ctfd", normed_complex_spectral, weight)
        proj = np.abs(np.einsum("ctfd, caf->dta", masked_angle, selected_steermat.conj()))
        gcc_nmf = tf.one_hot(np.argmax(proj, axis=-1), depth=len(outputs)).numpy()
        recon_spectral = np.einsum("dt, dta->dta", h, gcc_nmf)
        recon_spectral = np.einsum("fd, dta->atf", w, recon_spectral)
    elif method is "GDS_MENUT":
        proj = np.abs(np.einsum("ctf, caf->atf", normed_complex_spectral, selected_steermat.conj()))
        _, t, f = proj.shape
        proj = tf.one_hot(np.argmax(proj, axis=0), depth=len(outputs)).numpy()
        proj = tf.image.resize(tf.image.resize(proj, [t//2, f//2],
                                               tf.image.ResizeMethod.NEAREST_NEIGHBOR),
                               [t, f], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        plt.imshow(proj[..., 0])
        plt.show()
        plt.imshow(proj[..., 1])
        plt.show()
        recon_spectral = np.einsum("tfa, tf->atf", proj, spectral[ref_index])
    else:
        raise ValueError("No method is adopted, just print the angle!")
    recon_spectral = recon_spectral * normed_complex_spectral[ref_index][np.newaxis]
    recon_spectral = np.pad(recon_spectral, ((0, 0), (0, 0), (drop_band_lower, drop_band_upper))).astype(np.complex64)
    recon = tf.signal.inverse_stft(recon_spectral, frame_length=fft_length, frame_step=step,
                                   fft_length=fft_length).numpy()
    for i in range(len(recon)):
        wavfile.write(prefix + "/GCC_NMF/recon{}.wav".format(i), rate=8000, data=recon[i] / np.max(recon[i]))
