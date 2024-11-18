# pearCreateFile: DreamSound Studio/AST/ast_module.py
import tensorflow as tf
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt

def apply_low_pass_filter(audio, sample_rate, cutoff_frequency=8000, order=5):
    nyquist = 0.5 * sample_rate
    normalized_cutoff = cutoff_frequency / nyquist
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)
    smoothed_audio = filtfilt(b, a, audio)
    return smoothed_audio

# Constants
N_FFT = 1024
N_FILTERS = 4096
ALPHA = 1e-2/2
BETA = 5
learning_rate = 1e-1
iterations = 100

def read_audio_spectrum(filename):
    x, fs = librosa.load(filename, sr=None)
    print("Sampling rate:", fs)
    S = librosa.stft(x, n_fft=N_FFT)
    print("Shape of S:", S.shape)
    p = np.angle(S)
    integer = int(S.shape[1]*3/4)
    S = np.log1p(np.abs(S[:, : integer]))
    return S, fs

def build_model(x, kernel1, kernel2):
    kernel1_tf = tf.constant(kernel1)
    conv1 = tf.nn.conv2d(x, kernel1_tf, strides=[1, 1, 1, 1], padding="VALID")
    relu1 = tf.nn.relu(conv1)

    kernel2_tf = tf.constant(kernel2)
    conv2 = tf.nn.conv2d(relu1, kernel2_tf, strides=[1, 1, 1, 1], padding="VALID")
    relu2 = tf.nn.relu(conv2)

    return relu2

def audio_style_transfer(content_audio_path, style_audio_path):
    # Load content and style audio
    a_content, fs = read_audio_spectrum(content_audio_path)
    a_style, fs = read_audio_spectrum(style_audio_path)

    # Setup audio shapes
    N_SAMPLES = a_content.shape[1]
    N_CHANNELS = a_content.shape[0]
    a_style = a_style[:N_CHANNELS, :N_SAMPLES]
    a_content_tf = np.ascontiguousarray(a_content.T[None, None, :, :])
    a_style_tf = np.ascontiguousarray(a_style.T[None, None, :, :])

    # Initialize kernels
    std = np.sqrt(2) * np.sqrt(2.0 / ((N_CHANNELS + N_FILTERS) * 11))
    kernel1 = np.random.randn(1, 5, N_CHANNELS, N_FILTERS).astype(np.float32) * std
    kernel2 = np.random.randn(1, 5, N_FILTERS, N_FILTERS).astype(np.float32) * std

    # Convert inputs to tensors
    x_content = tf.convert_to_tensor(a_content_tf, dtype=tf.float32)
    x_style = tf.convert_to_tensor(a_style_tf, dtype=tf.float32)

    # Build models
    net_content = build_model(x_content, kernel1, kernel2)
    net_style = build_model(x_style, kernel1, kernel2)

    # Calculate features
    content_features = net_content
    features = tf.reshape(net_style, (-1, N_FILTERS))
    style_gram = tf.matmul(tf.transpose(features), features) / N_SAMPLES

    # Initialize generated audio
    x_gen = tf.Variable(tf.random.normal([1, 1, N_SAMPLES, N_CHANNELS]) * 1e-3, dtype=tf.float32)

    # Optimizer
    opt = tf.optimizers.Adam(learning_rate=learning_rate)

    # Training loop
    for i in range(iterations):
        with tf.GradientTape() as tape:
            net_gen = build_model(x_gen, kernel1, kernel2)
            content_loss = ALPHA * 2 * tf.nn.l2_loss(net_gen - content_features)
            feats = tf.reshape(net_gen, (-1, N_FILTERS))
            gram = tf.matmul(tf.transpose(feats), feats) / N_SAMPLES
            style_loss = BETA * 2 * tf.nn.l2_loss(gram - style_gram)
            loss = content_loss + style_loss
            
        gradients = tape.gradient(loss, [x_gen])
        opt.apply_gradients(zip(gradients, [x_gen]))
        
        if (i + 1) % 10 == 0:
            print(f"Iteration {i + 1}, Loss: {loss.numpy()}")

    # Convert result back to audio
    result = x_gen.numpy()
    a = np.zeros_like(a_content)
    a[:N_CHANNELS, :] = np.expm1(result[0, 0].T)

    # Phase reconstruction
    p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
    for _ in range(1000):
        S = a * np.exp(1j * p)
        x_rec = librosa.istft(S)
        p = np.angle(librosa.stft(x_rec, n_fft=N_FFT))

    x_rec_smoothed = apply_low_pass_filter(x_rec, fs)
    
    # Save the output
    output_path = "output_styled.wav"
    sf.write(output_path, x_rec_smoothed, fs)
    return x_rec_smoothed, fs
