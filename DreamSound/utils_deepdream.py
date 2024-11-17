
import os
import numpy as np
from io import BytesIO
from functools import partial
import PIL.Image
from IPython.display import clear_output, Image, display, HTML
import IPython.display
import tensorflow as tf
import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()
import librosa
import matplotlib.pyplot as plt
import soundfile as sf

# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def showarray(a, fmt='jpeg'):
    # Convert tensor to numpy if needed
    if isinstance(a, tf.Tensor):
        a = a.numpy() if tf.executing_eagerly() else sess.run(a)
    
    a = np.uint8(np.clip(a, 0, 1)*255)
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))

def visstd(a, s=0.1):
    '''Normalize the image range for visualization'''
    return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5

def load_inception_model():
    # Use absolute path for inception model
    model_fn = os.path.join(PROJECT_ROOT, 'Inception_graph', 'tensorflow_inception_graph.pb')
    print(f"Loading inception model from: {model_fn}")
    
    if not os.path.exists(model_fn):
        raise FileNotFoundError(f"Inception model not found at {model_fn}")
        
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

def T(layer):
    '''Helper for getting layer output tensor'''
    return graph.get_tensor_by_name(f"import/{layer}:0")



def calc_grad_tiled(img, t_grad, tile_size=512):
    sz = tile_size
    
    if isinstance(img, tf.Tensor):
        img = img.numpy() if tf.executing_eagerly() else sess.run(img)
    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    
    img_tf = tf.convert_to_tensor(img)
    img_shift = tf.roll(tf.roll(img_tf, shift=sx, axis=1), shift=sy, axis=0)
    img_shift_np = sess.run(img_shift)
    
    grad = np.zeros(img.shape, dtype=img.dtype)
    
    for y in range(0, max(h-sz//2, sz), sz):
        for x in range(0, max(w-sz//2, sz), sz):
            y_end = min(y + sz, h)
            x_end = min(x + sz, w)
            
            sub = img_shift_np[y:y_end, x:x_end]
            g = sess.run(t_grad, {t_input: sub})
            
            g_h, g_w = g.shape[:2]
            grad[y:y+g_h, x:x+g_w] = g
    
    grad_tf = tf.convert_to_tensor(grad)
    grad_shifted = tf.roll(tf.roll(grad_tf, shift=-sx, axis=1), shift=-sy, axis=0)
    return sess.run(grad_shifted)

# Laplacian Pyramid utilities
k = np.float32([1,4,6,4,1])
k = np.outer(k, k)
k5x5 = k[:,:,None,None]/k.sum()*np.eye(3, dtype=np.float32)

def lap_split(img):
    with tf1.name_scope('split'):
        lo = tf1.nn.conv2d(img, k5x5, [1,2,2,1], 'SAME')
        lo2 = tf1.nn.conv2d_transpose(lo, k5x5*4, tf1.shape(img), [1,2,2,1])
        hi = img-lo2
    return lo, hi

def lap_split_n(img, n):
    levels = []
    for i in range(n):
        img, hi = lap_split(img)
        levels.append(hi)
    levels.append(img)
    return levels[::-1]

def lap_merge(levels):
    img = levels[0]
    for hi in levels[1:]:
        with tf1.name_scope('merge'):
            img = tf1.nn.conv2d_transpose(img, k5x5*4, tf1.shape(hi), [1,2,2,1]) + hi
    return img

def normalize_std(img, eps=1e-10):
    with tf1.name_scope('normalize'):
        std = tf1.sqrt(tf1.reduce_mean(tf1.square(img)))
        return img/tf1.maximum(std, eps)

def lap_normalize(img, scale_n=4):
    img = tf1.expand_dims(img,0)
    tlevels = lap_split_n(img, scale_n)
    tlevels = list(map(normalize_std, tlevels))
    out = lap_merge(tlevels)
    return out[0,:,:,:]

def render_deepdream(t_obj, img0, iter_n=15, step=1.5, octave_n=8, octave_scale=1.4):
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
            if isinstance(g, tf.Tensor):
                g = g.numpy() if tf.executing_eagerly() else sess.run(g)
            if isinstance(img, tf.Tensor):
                img = img.numpy() if tf.executing_eagerly() else sess.run(img)
            
            img += g*(step / (np.abs(g).mean()+1e-7))
            print('.', end=' ')
        clear_output()
        if isinstance(img, tf.Tensor):
            img = img.numpy() if tf.executing_eagerly() else sess.run(img)
        showarray(img/255.0)
    return img/255.0

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

# Initialize model globally
try:
    graph, sess, t_input = load_inception_model()
except Exception as e:
    print(f"Error loading inception model: {str(e)}")
    raise
