import tensorflow as tf
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt


content_audio_path = "../audio/fade.mp3"
style_audio_path = "../audio/dont.mp3"
output_audio_path = "../audio/outdont2.wav"


def apply_low_pass_filter(audio, sample_rate, cutoff_frequency=8000, order=5):
    """
    Applies a low-pass Butterworth filter to smooth the audio signal.
    """
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

# Reads wav file and produces spectrum (Fourier phases are ignored)
def read_audio_spectrum(filename):
    x, fs = librosa.load(filename, sr=None)
    print("Sampling rate:", fs)
    S = librosa.stft(x, n_fft=N_FFT)
    print("Shape of S:", S.shape)
    p = np.angle(S)
    integer = int(S.shape[1]*3/4)
    S = np.log1p(np.abs(S[:, : integer]))  # Log amplitude spectrum
    return S, fs



def build_model(x, kernel1, kernel2):
    # First convolutional layer with kernel1
    kernel1_tf = tf.constant(kernel1)
    conv1 = tf.nn.conv2d(x, kernel1_tf, strides=[1, 1, 1, 1], padding="VALID")
    relu1 = tf.nn.relu(conv1)

    # Second convolutional layer with kernel2
    kernel2_tf = tf.constant(kernel2)
    conv2 = tf.nn.conv2d(relu1, kernel2_tf, strides=[1, 1, 1, 1], padding="VALID")
    relu2 = tf.nn.relu(conv2)

    return relu2

def audio_style_transfer(content_audio_path, style_audio_path):
    # Load content and style audio
    a_content, fs = read_audio_spectrum(content_audio_path)
    a_style, fs = read_audio_spectrum(style_audio_path)

    # Setup audio shapes and kernel
    N_SAMPLES = a_content.shape[1]
    N_CHANNELS = a_content.shape[0]
    a_style = a_style[:N_CHANNELS, :N_SAMPLES]
    a_content_tf = np.ascontiguousarray(a_content.T[None, None, :, :])
    a_style_tf = np.ascontiguousarray(a_style.T[None, None, :, :])

    # Initialize kernels with explicit float32 dtype
    std = np.float32(np.sqrt(2) * np.sqrt(2.0 / ((N_CHANNELS + N_FILTERS) * 11)))
    kernel1 = np.random.randn(1, 5, N_CHANNELS, N_FILTERS).astype(np.float32) * std
    kernel2 = np.random.randn(1, 5, N_FILTERS, N_FILTERS).astype(np.float32) * std


    # Filter shape "[filter_height, filter_width, in_channels, out_channels]"
    std = np.sqrt(2) * np.sqrt(2.0 / ((N_CHANNELS + N_FILTERS) * 11))

    # Define content and style features
    x_content = tf.convert_to_tensor(a_content_tf, dtype=tf.float32)
    x_style = tf.convert_to_tensor(a_style_tf, dtype=tf.float32)
    net_content = build_model(x_content, kernel1, kernel2)
    net_style = build_model(x_style, kernel1, kernel2)

    # Calculate content and style features
    content_features = net_content
    features = tf.reshape(net_style, (-1, N_FILTERS))
    style_gram = tf.matmul(tf.transpose(features), features) / N_SAMPLES

    # Optimizing the content and style features
    x_gen = tf.Variable(tf.random.normal([1, 1, N_SAMPLES, N_CHANNELS]) * 1e-3, dtype=tf.float32)

    # Build model with generated input
    net_gen = build_model(x_gen, kernel1, kernel2)
    content_loss = ALPHA * 2 * tf.nn.l2_loss(net_gen - content_features)

    # Style loss using Gram matrix
    feats = tf.reshape(net_gen, (-1, N_FILTERS))
    gram = tf.matmul(tf.transpose(feats), feats) / N_SAMPLES
    style_loss = BETA * 2 * tf.nn.l2_loss(gram - style_gram)

    # Total loss
    loss = content_loss + style_loss

    # Optimizer
    opt = tf.optimizers.Adam(learning_rate=learning_rate)

    # Training function
    @tf.function
    def train_step():
        with tf.GradientTape() as tape:
            net_gen = build_model(x_gen, kernel1, kernel2)
            content_loss = ALPHA * 2 * tf.nn.l2_loss(net_gen - content_features)
            feats = tf.reshape(net_gen, (-1, N_FILTERS))
            gram = tf.matmul(tf.transpose(feats), feats) / N_SAMPLES
            style_loss = BETA * 2 * tf.nn.l2_loss(gram - style_gram)
            loss = content_loss + style_loss
        gradients = tape.gradient(loss, [x_gen])
        opt.apply_gradients(zip(gradients, [x_gen]))
        return loss

    # Training loop
    for i in range(iterations):
        loss_value = train_step()
        if (i + 1) % 10 == 0:
            print(f"Iteration {i + 1}, Loss: {loss_value.numpy()}")

    # Extracting and converting the optimized result back to audio
    result = x_gen.numpy()
    a = np.zeros_like(a_content)
    a[:N_CHANNELS, :] = np.expm1(result[0, 0].T)

    # Phase reconstruction using Griffin-Lim
    p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
    for _ in range(1000):
        S = a * np.exp(1j * p)
        x_rec = librosa.istft(S)
        p = np.angle(librosa.stft(x_rec, n_fft=N_FFT))

    x_rec_smoothed = apply_low_pass_filter(x_rec, fs, cutoff_frequency=8000, order=5)
    # Saving the output
    sf.write(output_audio_path, x_rec_smoothed, fs)
    print(f"Output audio saved to {output_audio_path}")



def plot_spectrograms(content_audio_path, style_audio_path, output_audio_path):
    """Utility function to plot spectrograms"""
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
    return fig


# Only run this if the script is run directly
if __name__ == "__main__":
    print("Starting audio style transfer...")
    audio_style_transfer(content_audio_path, style_audio_path)
    print("Audio style transfer completed!")
