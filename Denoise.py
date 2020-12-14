import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as k
import librosa
import tensorflow as tf
import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
from sklearn.decomposition import NMF


if __name__ == '__main__':
    ref_index = 0
    fs = 8000
    r = 0.02
    fft_length = 512
    step = 128
    drop_band_lower = 10
    drop_band_upper = 10
    prefix = "/home/jax/Desktop/datas2/DATA_water2"
    with open(prefix + "/DATA.TXT", "rb") as f:
        recv_signals = f.read()
    recv_signals = tf.io.decode_raw(recv_signals, tf.int32)
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
    spatial_power = gcc_map * (power**2)[ref_index][np.newaxis]
    spatial_power = np.sum(np.sum(spatial_power, axis=-1), axis=-1)
    spatial_power /= np.max(spatial_power)
    outputs = np.argmax(spatial_power)
    selected_steermat = steermat[:, outputs]
    nmf = NMF(n_components=40, alpha=0.1, beta_loss="kullback-leibler", max_iter=600, tol=1e-2, solver="mu")
    w = nmf.fit_transform(power[ref_index].T)
    h = nmf.components_
    weight = w / np.sqrt(np.sum(w**2, axis=0))
    masked_angle = np.einsum("ctf, fd->ctfd", normed_complex_spectral, weight)
    proj = np.abs(np.einsum("ctfd, cf->dtf", masked_angle, selected_steermat.conj()))
    count, count_range = np.histogram(proj.reshape([-1]), bins=100)
    threhold = count_range[5]
    gcc_nmf = proj > threhold
    recon_spectral = np.einsum("dt, dtf->dtf", h, gcc_nmf)
    recon_spectral = np.einsum("fd, dtf->tf", w, recon_spectral)
    recon_spectral = recon_spectral * normed_complex_spectral[ref_index]
    recon_spectral = np.pad(recon_spectral, ((0, 0), (drop_band_lower, drop_band_upper))).astype(np.complex64)
    recon = tf.signal.inverse_stft(recon_spectral, frame_length=fft_length, frame_step=step,
                                   fft_length=fft_length).numpy()
    plt.plot(recv_signals[0])
    plt.show()
    plt.plot(recon)
    plt.show()
    wavfile.write(prefix + "/GCC_NMF/recon.wav", rate=8000, data=recon / np.max(recon))
