import matplotlib.pyplot as plt
from tensorflow.python.keras import backend as k
import librosa
import tensorflow as tf
import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.linalg import eigh
from pyroomacoustics.bss.common import projection_back

if __name__ == '__main__':
    prefix = "/home/jax/Desktop/datas2/DATA4"
    with open(prefix+"/DATA.TXT", "rb") as f:
        recv_signals = f.read()
    recv_signals = tf.io.decode_raw(recv_signals, tf.int32)
    recv_signals = tf.reshape(recv_signals, [-1, 8]).numpy().T[:-1, 8000*2:-8000*1].astype(np.float)
    recv_signals -= np.mean(recv_signals)
    recv_signals /= np.max(np.abs(recv_signals))
    # recv_signals = librosa.resample(recv_signals, orig_sr=45000, target_sr=8000)
    spectral = tf.signal.stft(recv_signals, frame_length=1024, frame_step=256, fft_length=1024).numpy()
    power = np.abs(spectral)
    low_power_mask = power[0] > 1e-2 * np.max(power[0])
    attention_weight = np.log(power[np.newaxis]+1e-6) - np.log(power[:, np.newaxis]+1e-6)
    attention_weight = np.sum(np.sum(attention_weight, axis=-1), axis=-1)
    attention_weight /= np.max(attention_weight)
    # Amp processing
    angle = np.angle(spectral)
    phase_features = np.sin(angle[np.newaxis] - angle[:, np.newaxis])
    phase_features -= np.mean(phase_features, axis=1, keepdims=True)
    phase_features /= np.maximum(np.std(phase_features, axis=1, keepdims=True), 1e-8)
    features = phase_features * attention_weight[..., np.newaxis, np.newaxis]
    # Point Level normalize
    _, _, t, f = features.shape
    low_power_mask = low_power_mask.reshape([t * f])
    features = np.reshape(features, [_ ** 2, t * f]).T
    features = features[low_power_mask]
    out = KMeans(n_clusters=2).fit_transform(features)
    # Drop Low similar point
    dropout = 0.4
    count, data_range = np.histogram(out.reshape([-1]), bins=100)
    count = count / out.reshape([-1]).shape[0]
    target_index = 0
    accumulator = 0
    for i in count:
        if not accumulator > dropout:
            target_index += 1
            accumulator += i
        else:
            break
    threhold = np.mean(data_range[target_index:target_index + 1])
    mask = np.clip((out[..., 0] > threhold) + (out[..., 1] > threhold), a_min=0., a_max=1.)[..., np.newaxis]
    out *= mask
    accumulator = np.zeros([t * f, 2])
    accumulator[low_power_mask] = out
    accumulator = np.reshape(accumulator, [t, f, 2])
    est_mask = tf.one_hot(np.argmax(accumulator, axis=-1), depth=2, dtype=tf.float32).numpy()
    est_mask *= low_power_mask.reshape([t, f, 1])
    normed_spectral = np.exp(1j*np.angle(spectral))
    ryy = np.einsum("itf, jtf->tfij", normed_spectral, normed_spectral.conj())
    ryy *= np.abs(spectral)[0, ..., np.newaxis, np.newaxis]
    # ryy = np.log1p(np.abs(ryy)+1e-9) * np.exp(1j*np.angle(ryy))
    rxx = np.einsum("tfa, tfij->afij", est_mask, ryy)
    ryy = np.sum(ryy, axis=0)
    spk, freq, chs, _ = rxx.shape
    for i in range(spk):
        speech_r = rxx[i]
        wgev_mvdr = []
        for j in range(freq):
            try:
                eigvals, eigvecs = eigh(speech_r[j], ryy[j]-speech_r[j], eigvals_only=False)
                wgev_mvdr.append(eigvecs[..., -1])
            except:
                wgev_mvdr.append(np.zeros([chs]))
        wgev_mvdr = np.asarray(wgev_mvdr)
        recon_spectral = np.einsum("itf, fi->tf", spectral, wgev_mvdr.conj())
        recon_speech = tf.signal.inverse_stft(recon_spectral, frame_length=1024, frame_step=256, fft_length=1024).numpy()
        wavfile.write(prefix+"/MENUT_drivenGEV/{}.wav".format(i), rate=8000, data=recon_speech/np.max(recon_speech))
