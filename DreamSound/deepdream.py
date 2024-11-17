import os
import numpy as np
import matplotlib.pyplot as plt
import IPython.display
import librosa
import soundfile as sf
from utils_deepdream import (
    process_audio,
    render_deepdream,
    resynthesize_audio,
    T,
    PROJECT_ROOT
)

def main():
    # Use absolute path for audio file
    audio_path = os.path.join(PROJECT_ROOT, 'audio', 'thief-original.aiff')
    print(f"Loading audio from: {audio_path}")
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at {audio_path}")
    
    img, orig_mag, orig_ang, sr = process_audio(audio_path)
    
    # Show original spectrogram      
    plt.figure(figsize=(12, 8))
    librosa.display.specshow(np.log(orig_mag), sr=sr, x_axis='time', y_axis='log')
    plt.title('Original Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.show()
    
    # Apply deep dream
    layer = 'mixed3b_1x1_pre_relu'
    channel = 13
    dream_spec = render_deepdream(T(layer)[:,:,:,channel], img)
    
    # Resynthesize audio
    output = resynthesize_audio(dream_spec, orig_mag, orig_ang, sr)
    
    # Play the result (if in notebook environment)
    # IPython.display.Audio(data=output, rate=sr)
    
    # Save the result
    output_path = os.path.join(PROJECT_ROOT, 'audio', 'output_dream_sound.wav')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, output, sr)

if __name__ == "__main__":
    main()
