# pearCreateFile: DreamSound Studio/DreamSound/deep_dream_module.py
import tensorflow as tf
import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()
import numpy as np
import librosa
import PIL.Image
from io import BytesIO
import soundfile as sf
import os

def load_inception_model():
    # Get the absolute path to the Inception model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    model_fn = os.path.join(parent_dir, 'Inception_graph', 'tensorflow_inception_graph.pb')
    
    # Check if file exists
    if not os.path.exists(model_fn):
        raise FileNotFoundError(f"Inception model not found at: {model_fn}")
    
    graph = tf1.Graph()
    sess = tf1.InteractiveSession(graph=graph)
    with tf.io.gfile.GFile(model_fn, 'rb') as f:
        graph_def = tf1.GraphDef()
        graph_def.ParseFromString(f.read())
    t_input = tf1.placeholder(np.float32, name='input')
    imagenet_mean = 117.0
    t_preprocessed = tf1.expand_dims(t_input-imagenet_mean, 0)
    tf1.import_graph_def(graph_def, {'input': t_preprocessed})
    return graph, sess, t_input

def process_audio(audio_path, sr=44100):
    y, sr = librosa.load(audio_path, sr=sr)
    nfft = 2048
    hop = 256
    y_stft = librosa.stft(y, n_fft=nfft, hop_length=hop, center=True)
    y_stft_mag, y_stft_ang = librosa.magphase(y_stft)
    nonlin = 1.0/8.0
    y_stft_mag_scaled = np.power(y_stft_mag, nonlin)
    y_stft_mag_scaled = np.flipud((1 - y_stft_mag_scaled/y_stft_mag_scaled.max()))
    y_stft_mag_rgb = np.stack([y_stft_mag_scaled]*3, axis=-1)
    img = 255 * y_stft_mag_rgb
    return img, y_stft_mag, y_stft_ang, sr

def resynthesize_audio(dream_spec, original_mag, original_ang, sr=44100):
    deepdream_out = np.flipud(dream_spec)
    deepdream_out = (1 - deepdream_out) * original_mag.max()
    deepdream_out = np.power(deepdream_out, 1/0.125)
    deepdream_out = np.sum(deepdream_out, axis=2) / 3.0
    deepdream_out = deepdream_out * original_ang
    output = librosa.istft(deepdream_out, hop_length=256, win_length=2048, center=True)
    return output

def calc_grad_tiled(img, t_grad, tile_size=512):
    sz = tile_size
    
    # First, ensure img is a numpy array
    if isinstance(img, tf.Tensor):
        img = img.numpy() if tf.executing_eagerly() else sess.run(img)
    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    
    # Convert to tensor and evaluate using session
    img_tf = tf.convert_to_tensor(img)
    
    img_shift = tf.roll(tf.roll(img_tf, shift=sx, axis=1), shift=sy, axis=0)
    img_shift_np = sess.run(img_shift)  # Convert back to numpy
    
    # Initialize grad as numpy array
    grad = np.zeros_like(img)
    
    # Use original tiling strategy
    for y in range(0, max(h-sz//2, sz), sz):
        for x in range(0, max(w-sz//2, sz), sz):
            y_end = min(y + sz, h)
            x_end = min(x + sz, w)
            
            sub = img_shift_np[y:y_end, x:x_end]
            g = sess.run(t_grad, {t_input: sub})
            
            # Handle potential size mismatch
            g_h, g_w = g.shape[:2]
            grad[y:y+g_h, x:x+g_w] = g
    
    # Convert to tensor only when needed for the roll operations
    grad_tf = tf.convert_to_tensor(grad)
    grad_shifted = tf.roll(tf.roll(grad_tf, shift=-sx, axis=1), shift=-sy, axis=0)
    return sess.run(grad_shifted)

def render_deepdream(t_obj, img0, iter_n=15, step=1.5, octave_n=8, octave_scale=1.4):
    global sess, t_input  # Access the global session and input tensor
    
    t_obj_scaled = tf1.multiply(t_obj, tf1.cast(tf1.math.log(t_obj) < 0.8*tf1.reduce_max(t_obj), tf.float32))
    t_score = tf1.reduce_mean(t_obj_scaled)
    t_grad = tf1.gradients(t_score, t_input)[0]
    
    img = img0.copy()
    octaves = []
    
    for i in range(octave_n-1):
        hw = img.shape[:2]
        lo = tf.image.resize(img[np.newaxis], np.int32(np.float32(hw)/octave_scale))[0]
        hi = img-tf.image.resize(lo[np.newaxis], hw)[0]
        img = lo
        octaves.append(hi)
    
    for octave in range(octave_n):
        if octave>0:
            hi = octaves[-octave]
            img = tf.image.resize(img[np.newaxis], hi.shape[:2])[0] + hi
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            # Convert g to numpy if it's a tensor
            if isinstance(g, tf.Tensor):
                g = g.numpy() if tf.executing_eagerly() else sess.run(g)
            # Convert img to numpy if it's a tensor
            if isinstance(img, tf.Tensor):
                img = img.numpy() if tf.executing_eagerly() else sess.run(img)
            
            img += g*(step / (np.abs(g).mean()+1e-7))
    
    return img/255.0

# Global variables for TensorFlow session and input
sess = None
t_input = None
graph = None

def apply_deep_dream(audio_path):
    global sess, t_input, graph
    try:
        # Initialize model
        graph, sess, t_input = load_inception_model()
        
        # Process audio
        img, orig_mag, orig_ang, sr = process_audio(audio_path)
        
        # Apply deep dream
        layer = 'mixed3b_1x1_pre_relu'
        channel = 13
        
        # Get layer tensor
        t_obj = graph.get_tensor_by_name(f"import/{layer}:0")[:,:,:,channel]
        
        # Apply deep dream
        dream_spec = render_deepdream(t_obj, img)
        
        # Resynthesize audio
        output = resynthesize_audio(dream_spec, orig_mag, orig_ang, sr)
        
        return output, sr
    
    except Exception as e:
        print(f"Error in deep dream processing: {str(e)}")
        raise
    finally:
        # Clean up TensorFlow session
        if sess is not None:
            sess.close()
