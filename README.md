## Vision-Verse

## Overview

Vision Verse is an AI-powered project that bridges the gap between visual imagery and poetic expression. Using ImageNet images as input and a transformer-based model trained on Gutenberg poetry, the system generates poems that reflect the mood, content, or theme of a given photograph. The project combines deep learning techniques from natural language processing (NLP) and computer vision to create artistic text outputs based on visual stimuli.

## Features

Image-to-Poetry Translation: Vision Verse takes an image as input and generates a poetic description, drawing inspiration from a vast corpus of poetry.

Transformer-Based Language Model: The project leverages a transformer architecture trained on Gutenberg poetry to produce meaningful and stylistic poetry.

Customizable Poem Length: The user can define how many words the generated poem should have.

Semantic Coherence: The generated poetry reflects the content or emotion of the input image.

## Objectives

The primary objective of Vision Verse is to explore the synergy between computer vision and natural language generation. By using AI to interpret visual data and convert it into poetic expressions, this project aims to enhance creative processes and offer new forms of artistic interpretation.

## Project Components
### 1. Poetry Model Training

The core model of Vision Verse is trained using a collection of poetry from the Gutenberg archive. This text corpus is used to train a sequential LSTM (Long Short-Term Memory) model that learns the structures and styles of poetic language.

Text Preprocessing: Poetry lines are tokenized and sequences of words are transformed into padded sequences to feed into the model.

Model Architecture: The model consists of multiple layers including embedding, bidirectional LSTM, and dense layers with a softmax output to predict the next word in the sequence.

Training: The model is trained using Gutenberg poetry, and early stopping and model checkpoints are applied to optimize performance.

### 2. Image Description to Poetic Generation

After training the model on poetic text, the next step is to convert image descriptions into poetic verses:

Text Generation: Using a seed text (typically an image description), the model generates a poetic response. This is done by predicting the next word in a sequence iteratively, thus generating coherent and stylistically appropriate poetry.

### 3. ImageNet Dataset Integration

Although the current implementation focuses on generating poetry based on textual descriptions, the next phase involves integrating ImageNet images to directly extract features from images and generate corresponding poetic text. This will be achieved through a vision transformer model to process the visual data and map it to poetic outputs.

## Usage
Poetry Generation from Text Description

To generate poetry based on a given description, you can call the generate_poetic_text() function with a seed description and the desired number of words.
