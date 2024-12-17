# pearCreateFile: DreamSound Studio/app.py
import gradio as gr
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from pathlib import Path
import os
import tempfile
import base64

from AST.ast_module import audio_style_transfer
from DreamSound.deep_dream_module import apply_deep_dream

def get_base64_images():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    images = {}
    
    # Hero background
    hero_path = os.path.join(current_dir, 'assets', 'hero-bg.jpg')
    if os.path.exists(hero_path):
        with open(hero_path, "rb") as image_file:
            images['hero'] = base64.b64encode(image_file.read()).decode()
    
    # Program section background
    program_path = os.path.join(current_dir, 'assets', 'program-bg.jpg')
    if os.path.exists(program_path):
        with open(program_path, "rb") as image_file:
            images['program'] = base64.b64encode(image_file.read()).decode()

    # About section background
    program_path = os.path.join(current_dir, 'assets', 'about-bg.jpg')
    if os.path.exists(program_path):
        with open(program_path, "rb") as image_file:
            images['about'] = base64.b64encode(image_file.read()).decode()

    # Gradio container background
    gradio_container_path = os.path.join(current_dir, 'assets', 'container-bg.jpg')
    if os.path.exists(gradio_container_path):
        with open(gradio_container_path, "rb") as image_file:
            images['gradio_container'] = base64.b64encode(image_file.read()).decode()
    
    return images

# Get base64 images
images = get_base64_images()
hero_bg = f"data:image/jpeg;base64,{images.get('hero', '')}"
program_bg = f"data:image/jpeg;base64,{images.get('program', '')}"
about_bg = f"data:image/jpeg;base64,{images.get('about', '')}"
container_bg = f"data:image/jpeg;base64,{images.get('gradio_container', '')}"

# Custom CSS with base64 backgrounds
custom_css = f"""
.gradio-container {{

    background-color: #1a1a1a;
    background: linear-gradient(rgba(42, 42, 42, 0.9), rgba(42, 42, 42, 0.9)),
                url('{program_bg}');
    color: #ffffff;
}}

.hero-section {{
    background-image: url('{hero_bg}');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    padding: 60px 0;
    text-align: center;
    margin-bottom: 40px;
    color: white;
    border-radius: 10px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    position: relative;
    width: 100%;
}}

.hero-section::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(rgba(42, 42, 42, 0.5), rgba(42, 42, 42, 0.5));
    border-radius: 10px;
    z-index: 1;
}}

.hero-section > * {{
    position: relative;
    z-index: 2;
}}

.title-container {{
    background: rgba(25, 25, 25, 0.1);  /* Similar to feature cards */
    border-radius: 15px;
    padding: 25px 40px;
    border: 1px solid #444;
    margin: 0 auto;  /* Centers the container */
    max-width: 800px;  /* Limits the width */
    position: relative;
    overflow: hidden;
}}

.title-container::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
}}

.site-title {{
    margin-bottom: 0;  /* Override previous margin */
}}

.site-title h1 {{
    font-size: 4.5em;
    margin: 0;
    color: #4ecdc4;
    text-shadow:
        2px 2px 5px rgba(0, 0, 0, 0.5),
        -2px -2px 5px rgba(255, 255, 255, 0.3);
    text-transform: uppercase;
    letter-spacing: 3px;
    font-weight: 800;
}}

.site-title p {{
    color: #cccccc;
    font-size: 1.2em;
    margin-top: 10px;
    font-style: italic;
    text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.5);
}}

.hero-content  {{
    margin-top: 60px;
}}

.hero-content h3 {{
    font-size: 2em;
    margin-bottom: auto;
}}

.hero-content p {{
    font-size: 1em;
    max-width: 800px;
    margin: auto;
}}

.about-section {{
    padding: 40px;
    background: linear-gradient(rgba(42, 42, 42, 0.5), rgba(42, 42, 42, 0.5)),
                url('{about_bg}');
    background-repeat: no-repeat;
    border-radius: 10px;
    margin: 20px 0;
}}

.about-section h2 {{
    text-align: center;
    font-size: 2.8em;
    margin-bottom: 30px;
    background: linear-gradient(45deg, #00ffd9, #ff1e1e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.feature-cards {{
    display: flex;
    gap: 30px;
    justify-content: center;
    align-items: stretch;
}}

.feature-card {{
    flex: 1;
    background: rgba(25, 25, 25, 0.8);
    border-radius: 15px;
    padding: 25px;
    border: 1px solid #444;
    transition: transform 0.3s, box-shadow 0.3s;
    position: relative;
    overflow: hidden;
}}

.feature-card::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
}}

.feature-card:hover {{
    transform: translateY(-5px);
    box-shadow: 0 5px 20px rgba(0, 255, 217, 0.2);
}}

.feature-card h3 {{
    font-size: 2em;
    margin-bottom: 20px;
    color: #fff;
    text-align: center;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
}}

.feature-card p {{
    color: #cccccc;
    line-height: 1.6;
    font-size: 1.1em;
    margin-bottom: 15px;
}}

.feature-card ul {{
    color: #cccccc;
    list-style-type: none;
    padding: 0;
    margin: 15px 0;
}}

.feature-card ul li {{
    padding: 8px 0;
    padding-left: 25px;
    position: relative;
}}

.feature-card ul li::before {{
    content: '✦';
    position: absolute;
    left: 0;
    color: #00ffd9;
}}

.program-section {{
    background: linear-gradient(rgba(42, 42, 42, 0.5), rgba(42, 42, 42, 0.5)),
                url('{about_bg}');
    background-size: cover;
    background-position: center;
    padding: 40px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
}}

.program-header {{
    text-align: center;
    margin-bottom: 40px;
    padding: 20px;
    border-bottom: 2px solid #444;
    background: rgba(30, 30, 30, 0.7);
}}

.program-header h2 {{
    font-size: 2.5em;
    margin-bottom: 15px;
    background: linear-gradient(45deg, #00ffd9, #ff1e1e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.creative-text {{
    font-size: 1.2em;
    color: #cccccc;
    max-width: 800px;
    margin: 0 auto;
    line-height: 1.6;
}}

.creative-workspace {{
    background: rgba(30, 30, 30, 0.7);
    border-radius: 10px;
    padding: 20px;
    gap: 30px;
}}

.section-header {{
    text-align: center;
    margin-bottom: 25px;
    padding: 15px;
    border-bottom: 1px solid #444;
}}

.section-header h3 {{
    font-size: 1.8em;
    margin-bottom: 10px;
    color: #4ecdc4;
}}

.section-header p {{
    color: #888;
    font-style: italic;
}}

.input-section, .output-section {{
    background: rgba(25, 25, 25, 0.7);
    border-radius: 8px;
    padding: 20px;
    border: 1px solid #333;
}}

.audio-input, .audio-output {{
    margin: 15px 0;
    padding: 10px;
    border: 1px solid #444;
    border-radius: 8px;
    background: rgba(35, 35, 35, 0.8);
}}

.input-description, .output-description {{
    text-align: center;
    font-size: 0.9em;
    color: #888;
    margin: 5px 0 20px 0;
    font-style: italic;
}}

.custom-button {{
    background: linear-gradient(45deg, #4ecdc4, #2c3e50);
    border: none;
    padding: 12px 25px;
    border-radius: 25px;
    color: white;
    font-weight: bold;
    cursor: pointer;
    transition: transform 0.2s, box-shadow 0.2s;
    margin: 15px 0;
    width: 100%;
}}

.custom-button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(78, 205, 196, 0.3);
}}

.dream-button {{
    background: linear-gradient(45deg, #ff6b6b, #c44e91);
}}

.dream-button:hover {{
    box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
}}

@keyframes glow {{
    0% {{ box-shadow: 0 0 5px rgba(78, 205, 196, 0.2); }}
    50% {{ box-shadow: 0 0 20px rgba(78, 205, 196, 0.4); }}
    100% {{ box-shadow: 0 0 5px rgba(78, 205, 196, 0.2); }}
}}

.audio-output {{
    animation: glow 3s infinite;
}}
"""

def process_audio_style_transfer(content_audio, style_audio):
    """Process audio files using Audio Style Transfer"""
    if content_audio is None or style_audio is None:
        return None
        
    # Save the input audio to temporary files
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as content_temp:
        sf.write(content_temp.name, content_audio[1], content_audio[0])
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as style_temp:
            sf.write(style_temp.name, style_audio[1], style_audio[0])
            print(content_temp.name, style_temp.name)
            
            # Process the audio files
            output_audio, sr = audio_style_transfer(content_temp.name, style_temp.name)
            
    # Clean up temporary files
    os.unlink(content_temp.name)
    os.unlink(style_temp.name)
    
    # Save the output
    output_path = os.path.join("audio", "outputs", "styled_output.wav")
    sf.write(output_path, output_audio, sr)
    return output_path

def process_deep_dream(audio):
    """Apply DeepDream to audio"""
    if audio is None:
        return None
        
    # Extract sample rate and audio data from Gradio's audio input
    sr, audio_data = audio
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
        sf.write(temp_audio.name, audio_data, sr)
        
        # Process the audio file
        output_audio, sr = apply_deep_dream(temp_audio.name)
        
    # Clean up temporary file
    os.unlink(temp_audio.name)
    
    # Save the output
    output_path = os.path.join("audio", "outputs", "dream_output.wav")
    sf.write(output_path, output_audio, sr)
    return output_path

# Create the Gradio interface
def create_ui():
    with gr.Blocks(css=custom_css) as app:
        # Hero Section with Site Title
        with gr.Row(elem_classes="hero-section"):
            gr.HTML("""
                <div class="title-container">
                    <div class="site-title">
                        <h1>DreamSound Studio</h1>
                        <p>Where AI Meets Musical Creativity</p>
                    </div>
                </div>
                <div class="hero-content">
                    <h2>Transform Your Music with AI</h2>
                    <p>Combine musical styles and create dream-like audio experiences</p>
                </div>
            """)
        
        # About Section
        with gr.Row(elem_classes="about-section"):
            gr.HTML("""
                <h2>About DreamSound Studio</h2>
                <div class="feature-cards">
                    <div class="feature-card">
                        <h3>Audio Style Transfer</h3>
                        <p>Experience the magic of neural style transfer applied to audio. This groundbreaking technology allows you to infuse the musical characteristics of one track into another.</p>
                        <ul>
                            <li>Preserve content structure while adopting new styles</li>
                            <li>Blend genres in ways never before possible</li>
                            <li>Create unique sonic signatures</li>
                            <li>Experiment with unlimited style combinations</li>
                        </ul>
                        <p>Perfect for musicians, producers, and sound designers looking to push creative boundaries and discover new sonic territories.</p>
                    </div>
                    
                    <div class="feature-card">
                        <h3>DeepDream Audio</h3>
                        <p>Dive into the surreal world of audio dreams. Using Google's DeepDream technology, reimagined for sound, create otherworldly audio experiences.</p>
                        <ul>
                            <li>Transform ordinary sounds into dreamlike soundscapes</li>
                            <li>Enhance and exaggerate sonic patterns</li>
                            <li>Generate unique audio textures</li>
                            <li>Explore AI-enhanced sound design</li>
                        </ul>
                        <p>An innovative tool for sound artists and experimental musicians seeking to explore the boundaries of audio manipulation.</p>
                    </div>
                </div>
            """)

        
        # Program Section
        with gr.Group(elem_classes="program-section"):
            gr.HTML("""
                <div class="program-header">
                    <h2>Create Your Dream Sound</h2>
                    <p class="creative-text">Transform your audio into something extraordinary. 
                    Mix styles, create surreal soundscapes, and push the boundaries of musical creativity.</p>
                </div>
            """)
            
            with gr.Row(elem_classes="creative-workspace"):
                # Input Column
                with gr.Column(elem_classes="input-section"):
                    gr.HTML("""
                        <div class="section-header">
                            <h3>Input Zone</h3>
                            <p>Start your creative journey here</p>
                        </div>
                    """)
                    content_input = gr.Audio(
                        label="Content Audio", 
                        type="numpy",
                        elem_classes="audio-input"
                    )
                    gr.HTML("""
                        <div class="input-description">
                            <p>↑ This is your base track. Choose something with a clear structure.</p>
                        </div>
                    """)
                    
                    style_input = gr.Audio(
                        label="Style Audio",
                        type="numpy",
                        elem_classes="audio-input"
                    )
                    gr.HTML("""
                        <div class="input-description">
                            <p>↑ This defines the sonic character. Be creative with your choice!</p>
                        </div>
                    """)
                    
                    process_btn = gr.Button(
                        "✨ Generate Style Transfer",
                        elem_classes="custom-button"
                    )
                
                # Output Column
                with gr.Column(elem_classes="output-section"):
                    gr.HTML("""
                        <div class="section-header">
                            <h3>Transformation Zone</h3>
                            <p>Watch your audio evolve</p>
                        </div>
                    """)
                    output_audio = gr.Audio(
                        label="Style Transfer Output",
                        elem_classes="audio-output"
                    )
                    gr.HTML("""
                        <div class="output-description">
                            <p>↑ Your fusion of content and style</p>
                        </div>
                    """)
                    
                    dream_btn = gr.Button(
                        "🌟 Apply DreamSound Magic",
                        elem_classes="custom-button dream-button"
                    )
                    
                    final_output = gr.Audio(
                        label="Final Dream Audio",
                        elem_classes="audio-output"
                    )
                    gr.HTML("""
                        <div class="output-description">
                            <p>↑ Your final dreamlike creation</p>
                        </div>
                    """)
            
            # Connect components
            process_btn.click(
                fn=process_audio_style_transfer,
                inputs=[content_input, style_input],
                outputs=output_audio
            )
            
            dream_btn.click(
                fn=process_deep_dream,
                inputs=[output_audio],
                outputs=final_output
            )
            
    return app

# Create necessary directories if they don't exist
current_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(current_dir, 'audio', 'outputs'), exist_ok=True)
os.makedirs(os.path.join(current_dir, 'assets'), exist_ok=True)

# Launch the app
if __name__ == "__main__":
    app = create_ui()
    app.launch()
