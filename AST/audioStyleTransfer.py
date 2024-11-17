import os
import numpy as np
import tensorflow as tf
import soundfile as sf
from utils_ast import (
    process_audio,
    build_model,
    initialize_kernels,
    train_step,
    reconstruct_audio,
    plot_spectrograms
)

def main():
    # Define paths
    content_audio_path = "../audio/fade.mp3"
    style_audio_path = "../audio/dont.mp3"
    output_audio_path = "../audio/outdont2.wav"
    
    # Process audio files
    a_content_tf, a_style_tf, a_content, N_CHANNELS, N_SAMPLES, fs = process_audio(
        content_audio_path, style_audio_path
    )
    
    # Initialize model components
    kernel1, kernel2 = initialize_kernels(N_CHANNELS)
    
    # Convert to tensors
    x_content = tf.convert_to_tensor(a_content_tf, dtype=tf.float32)
    x_style = tf.convert_to_tensor(a_style_tf, dtype=tf.float32)
    
    # Build model and compute features
    net_content = build_model(x_content, kernel1, kernel2)
    net_style = build_model(x_style, kernel1, kernel2)
    
    # Calculate content and style features
    content_features = net_content
    features = tf.reshape(net_style, (-1, N_FILTERS))
    global style_gram
    style_gram = tf.matmul(tf.transpose(features), features) / N_SAMPLES
    
    # Initialize generated audio
    x_gen = tf.Variable(
        tf.random.normal([1, 1, N_SAMPLES, N_CHANNELS]) * 1e-3, 
        dtype=tf.float32
    )
    
    # Training loop
    iterations = 100
    for i in range(iterations):
        loss_value = train_step(x_gen, kernel1, kernel2, content_features)
        if (i + 1) % 10 == 0:
            print(f"Iteration {i + 1}, Loss: {loss_value.numpy()}")
    
    # Reconstruct and save audio
    result = x_gen.numpy()
    output_audio = reconstruct_audio(result, a_content, fs)
    
    # Save the result
    sf.write(output_audio_path, output_audio, fs)
    print(f"Output audio saved to {output_audio_path}")
    
    # Plot spectrograms
    plot_spectrograms(content_audio_path, style_audio_path, output_audio_path)

if __name__ == "__main__":
    main()
