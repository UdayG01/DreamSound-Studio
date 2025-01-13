# DreamSound Studio - _AI Meets Musical Creativity_

Transform your audio with AI! DreamSound Studio combines the power of neural style transfer and DeepDream technology to create unique and dreamlike audio experiences.

<table>
  <tr>
    <td style="text-align: center;">
      <img src="https://github.com/user-attachments/assets/af8588af-6029-43f2-b6c5-50cba1365101" alt="DreamSound logo" style="width: 300px;">
    </td>
  </tr>
</table>

## Table of Contents

- [Introduction](#introduction)
- [Installation Instructions](#installation-instructions)
- [Usage Guide](#usage-guide)
- [Features](#features)
- [Architecture or Design](#architecture-or-design)
- [Workflow Diagram](#workflow-diagram)
- [Technologies and Dependencies](#technologies-and-dependencies)
- [Setup for Development](#setup-for-development)
- [Testing](#testing)
- [Known Issues and Roadmap](#known-issues-and-roadmap)
- [Acknowledgments](#acknowledgments)
- [Changelog](#changelog)
- [Contact Information](#contact-information)

## Introduction

DreamSound Studio is an innovative audio processing application that leverages advanced machine learning techniques to transform audio files in unprecedented ways. This project is designed for musicians, producers, sound designers, and audio enthusiasts who are looking to explore new creative possibilities in sound design.

### Overview

DreamSound Studio offers two main features: Audio Style Transfer and DeepDream Audio. Audio Style Transfer allows users to blend the musical characteristics of one audio track into another, creating a unique fusion of styles. DeepDream Audio, on the other hand, applies the surreal and hallucinogenic effects of Google's DeepDream to audio files, resulting in dreamlike and otherworldly soundscapes.

### Impact and Applications

DreamSound Studio has the potential to revolutionize the way we think about and interact with audio. By making advanced audio processing techniques accessible to a wider audience, it empowers users to create unique and innovative soundscapes that can be used in music production, film scoring, game development, and more. The project also serves as a valuable learning resource for those interested in the intersection of machine learning and audio processing.

- **Creative Sound Design**: Explore new creative possibilities in sound design by blending different musical styles and applying surreal effects to audio.
- **Practical Applications**: Use the transformed audio in various applications such as music production, soundtracks for films and games, and experimental audio art.
- **Educational Resource**: Gain insights into the implementation of machine learning models for audio processing, including data preparation, model training, and inference, and understanding how neural networks perceive audio data.



## Installation Instructions

### Prerequisites

- Python 3.8 or higher
- Virtual environment tool (e.g., `venv` or `virtualenv`)

### Steps

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/DreamSoundStudio.git
    cd DreamSoundStudio
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage Guide

### Running the Application

1. Start the application:
    ```sh
    python app.py
    ```

2. Open your web browser and navigate to the provided URL to access the DreamSound Studio interface.

### Using Audio Style Transfer

1. Upload your content audio file.
2. Upload your style audio file.
3. Click on "Generate Style Transfer" to blend the styles.

### Using DeepDream Audio

1. Upload your audio file.
2. Click on "Apply DreamSound Magic" to transform the audio into a dreamlike soundscape.

### Screenshots

<table>
  <tr>
    <td style="text-align: center;">
      <img src="https://github.com/user-attachments/assets/d4476e22-8f58-43bb-9112-89e0188f8ebb" alt="UI Screenshot 1" width="400">
    </td>
    <td style="text-align: center;">
      <img src="https://github.com/user-attachments/assets/b5ca3316-a61a-4071-8c46-4477aa4be473" alt="UI Screenshot 2" width="400">
    </td>
  </tr>
</table>



## Features

- **Audio Style Transfer**: Blend the musical characteristics of one track into another.
- **DeepDream Audio**: Create surreal and dreamlike audio experiences.
- **User-Friendly Interface**: Easy-to-use web interface powered by Gradio.

## Architecture or Design

### Project Structure

```plaintext
DreamSoundStudio/
├── app.py
├── requirements.txt
├── AST/
│   ├── ast_module.py
│   └── ...
├── DreamSound/
│   ├── deep_dream_module.py
│   └── ...
├── audio/
│   └── outputs/
└── assets/
    └── DreamSound Outputs/
```

## Flow of Execution

![Flow Chart](https://github.com/user-attachments/assets/741ff408-17a3-4048-ad7f-d2fc84f59e3c)

## Technologies and Dependencies

- **TensorFlow**: For machine learning models.
- **Librosa**: For audio processing.
- **Gradio**: For creating the web interface.
- **Matplotlib**: For visualizing spectrograms.

## Setup for Development

Follow the installation instructions to set up the environment.

Run the application in development mode:

```sh
python app.py
```

## Testing

Use the provided test scripts in the `Tests` directory to run unit tests.

## Known Issues and Roadmap

### Known Issues

- To fix how the Gradio UI can be adapted to use the tensorflow graph models
- Or use the Tensorflow's eager execution directly for creating the "Dream" effect

### Roadmap

- Add support for more audio formats.
- Improve the performance of style transfer.
- Add more customization options for DeepDream audio.

## Acknowledgments

- Thanks to the TensorFlow team for their amazing library.
- Inspired by Google's DeepDream project.
- [Thanks to this library providing me an in-depth understanding of using deep dream for audio](https://github.com/markostam/audio-deepdream-tf)
- [This library taught me the fundamentals of NST](https://github.com/rupeshs/neuralsongstyle)

### Version 1.0.0

- Initial release with audio style transfer and DeepDream audio features.


