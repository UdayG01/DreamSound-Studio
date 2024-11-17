import tensorflow as tf
import librosa
import numpy as np
import soundfile as sf
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Constants
N_FFT = 1024
N_FILTERS = 4096
ALPHA = 1e-2/2
BETA = 5

def apply_low_pass_filter(audio, sample_rate, cutoff_frequency=8000, order=5):
    """Applies a low-pass Butterworth filter to smooth the audio signal."""
    nyquist = 0.5 * sample_rate
    normalized_cutoff = cutoff_frequency / nyquist
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)
    smoothed_audio = filtfilt(b, a, audio)
    return smoothed_audio

def read_audio_spectrum(filename):
    """Reads wav file and produces spectrum (Fourier phases are ignored)."""
    x, fs = librosa.load(filename, sr=None)
    print("Sampling rate:", fs)
    S = librosa.stft(x, n_fft=N_FFT)
    print("Shape of S:", S.shape)
    p = np.angle(S)
    integer = int(S.shape[1]*3/4)
    S = np.log1p(np.abs(S[:, :integer]))  # Log amplitude spectrum
    return S, fs

def plot_spectrograms(content_audio_path, style_audio_path, output_audio_path):
    """Plots spectrograms of content, style, and output audio."""
    content_spec, _ = read_audio_spectrum(content_audio_path)
    style_spec, _ = read_audio_spectrum(style_audio_path)
    output_spec, _ = read_audio_spectrum(output_audio_path)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.imshow(content_spec, aspect='auto')
    ax1.set_title('Content Audio')
    
    ax2.imshow(style_spec, aspect='auto')
    ax2.set_title('Style Audio')
    
    ax3.imshow(output_spec, aspect='auto')
    ax3.set_title('Output Audio')
    
    plt.tight_layout()
    plt.show()

def build_model(x, kernel1, kernel2):
    """Builds the neural network model."""
    kernel1_tf = tf.constant(kernel1)
    conv1 = tf.nn.conv2d(x, kernel1_tf, strides=[1, 1, 1, 1], padding="VALID")
    relu1 = tf.nn.relu(conv1)

    kernel2_tf = tf.constant(kernel2)
    conv2 = tf.nn.conv2d(relu1, kernel2_tf, strides=[1, 1, 1, 1], padding="VALID")
    relu2 = tf.nn.relu(conv2)

    return relu2

def initialize_kernels(n_channels):
    """Initialize model kernels."""
    std = np.sqrt(2) * np.sqrt(2.0 / ((n_channels + N_FILTERS) * 11))
    kernel1 = np.random.randn(1, 5, n_channels, N_FILTERS).astype(np.float32) * std
    kernel2 = np.random.randn(1, 5, N_FILTERS, N_FILTERS).astype(np.float32) * std
    return kernel1, kernel2

@tf.function
def train_step(x_gen, kernel1, kernel2, content_features):
    """Single training step for style transfer."""
    with tf.GradientTape() as tape:
        net_gen = build_model(x_gen, kernel1, kernel2)
        content_loss = ALPHA * 2 * tf.nn.l2_loss(net_gen - content_features)
        feats = tf.reshape(net_gen, (-1, N_FILTERS))
        gram = tf.matmul(tf.transpose(feats), feats) / feats.shape[0]
        style_loss = BETA * 2 * tf.nn.l2_loss(gram - style_gram)
        loss = content_loss + style_loss
    gradients = tape.gradient(loss, [x_gen])
    opt.apply_gradients(zip(gradients, [x_gen]))
    return loss

def process_audio(content_path, style_path):
    """Process audio files and prepare for style transfer."""
    # Load and process audio files
    a_content, fs = read_audio_spectrum(content_path)
    a_style, fs = read_audio_spectrum(style_path)
    
    N_SAMPLES = a_content.shape[1]
    N_CHANNELS = a_content.shape[0]
    
    # Adjust style audio to match content dimensions
    a_style = a_style[:N_CHANNELS, :N_SAMPLES]
    
    # Convert to tensorflow format
    a_content_tf = np.ascontiguousarray(a_content.T[None, None, :, :])
    a_style_tf = np.ascontiguousarray(a_style.T[None, None, :, :])
    
    return a_content_tf, a_style_tf, a_content, N_CHANNELS, N_SAMPLES, fs

def reconstruct_audio(result, a_content, fs, n_fft=N_FFT):
    """Reconstruct audio from generated result."""
    a = np.zeros_like(a_content)
    a[:a_content.shape[0], :] = np.expm1(result[0, 0].T)
    
    # Phase reconstruction using Griffin-Lim
    p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
    for _ in range(1000):
        S = a * np.exp(1j * p)
        x_rec = librosa.istft(S)
        p = np.angle(librosa.stft(x_rec, n_fft=n_fft))
    
    # Apply smoothing
    x_rec_smoothed = apply_low_pass_filter(x_rec, fs)
    return x_rec_smoothed

# Initialize global variables
opt = tf.optimizers.Adam(learning_rate=1e-1)
style_gram = None  # Will be set during training
