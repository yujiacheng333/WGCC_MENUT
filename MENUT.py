import librosa
import tensorflow as tf
import numpy as np
from scipy.io import wavfile
from sklearn.cluster import KMeans
from pyroomacoustics.bss.common import projection_back


if __name__ == '__main__':
    spk = 2
    prefix = "/home/jax/Desktop/datas2/DATA4"
    with open(prefix + "/DATA.TXT", "rb") as f:
        recv_signals = f.read()
    recv_signals = tf.io.decode_raw(recv_signals, tf.int32)
    recv_signals = tf.reshape(recv_signals, [-1, 8]).numpy().T[:-1, 8000*2:-8000*1].astype(np.float)
    recv_signals -= np.mean(recv_signals)
    recv_signals /= np.max(np.abs(recv_signals))
    wavfile.write(prefix+"/noisy.wav", rate=8000, data=recv_signals[-1])
    spectral = tf.signal.stft(recv_signals, frame_length=1024, frame_step=256, fft_length=1024).numpy()
    power = np.abs(spectral)
    low_power_mask = power[0] > 1e-2 * np.max(power[0])
    attention_weight = np.log(power[np.newaxis]+1e-6) - np.log(power[:, np.newaxis]+1e-6)
    attention_weight = np.sum(np.sum(attention_weight, axis=-1), axis=-1)
    attention_weight /= np.max(attention_weight)
    # Amp processing
    angle = np.angle(spectral)
    phase_features = (angle[np.newaxis] - angle[:, np.newaxis]-np.pi)%(np.pi*2)
    phase_features -= np.mean(phase_features, axis=1, keepdims=True)
    phase_features /= np.maximum(np.std(phase_features, axis=1, keepdims=True), 1e-8)
    features = phase_features * attention_weight[..., np.newaxis, np.newaxis]
    # Point Level normalize
    _, _, t, f = features.shape
    low_power_mask = low_power_mask.reshape([t * f])
    features = np.reshape(features, [_**2, t * f]).T
    features = features[low_power_mask]
    out = KMeans(n_clusters=spk).fit_transform(features)
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
    threhold = np.mean(data_range[target_index:target_index+1])
    mask = np.clip((out[..., 0] > threhold) + (out[..., 1] > threhold), a_min=0., a_max=1.)[..., np.newaxis]
    out *= mask
    accumulator = np.zeros([t * f, spk])
    accumulator[low_power_mask] = out
    accumulator = np.reshape(accumulator, [t, f, spk])
    out = accumulator
    out = tf.one_hot(np.argmax(out, axis=-1), depth=spk, dtype=tf.float32).numpy()
    # out = tf.nn.softmax(out * 5, axis=-1).numpy()
    recon = out * spectral[0, ..., np.newaxis]
    z = projection_back(recon, spectral[0])
    recon = recon * np.conj(z[np.newaxis, :, :])
    recon = recon.transpose([2, 0, 1])
    recon = tf.signal.inverse_stft(recon, frame_length=1024, frame_step=256, fft_length=1024)
    for i in range(spk):
        wavfile.write(prefix+"/MENUT/recon{}.wav".format(i), rate=8000, data=recon[i].numpy() / np.max(recon[i]))
