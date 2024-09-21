## Vision-Verse

## Overview

Vision Verse is an innovative AI project that generates poetry from visual images by using a combination of transformer-based models. Specifically, the project employs a Vision Transformer (ViT) as the encoder and GPT-2 as the decoder. By fine-tuning these models on both image datasets (Flickr30k) and text-based datasets (Project Gutenberg poetry), the system is capable of generating creative poetic descriptions from photographs.

## Features

Image-to-Poetry Generation: Vision Verse interprets an image and creates a unique poem that encapsulates the scene, mood, or theme of the visual input.

Transformer-Based Architecture: The system combines a Vision Transformer (ViT) for image feature extraction and GPT-2 for poetry generation.

Flexible and Dynamic Output: The user can customize the number of words or lines in the generated poem, creating personalized outputs based on the input image.

## Objectives

Vision Verse aims to demonstrate the power of AI in bridging the gap between vision and language. The project showcases how deep learning can be used creatively to interpret visual data and convert it into artistic forms of expression like poetry.

## Project Components
### 1. Model Architecture

Encoder: Vision Transformer (ViT) - A powerful model for extracting visual features from images.

Decoder: GPT-2 - A pre-trained language model that generates coherent and stylistic poetry based on the visual input.

Feature Extractor: ViTFeatureExtractor is used to transform images into feature vectors, capturing the necessary visual information.

Tokenizer: GPT-2 is tokenized using the AutoTokenizer from Hugging Face, converting text into numerical sequences for processing.

### 2. Datasets

Flickr30k: A widely-used image dataset consisting of 31,783 images. Each image is accompanied by five human-annotated descriptive captions, capturing a wide range of everyday activities and scenes.

Project Gutenberg Poetry Dataset: Contains 3,085,117 lines of poetry from hundreds of books, each line corresponding to a unique Gutenberg ID. This text corpus provides the necessary training material to teach GPT-2 the structure and language of poetry.

### 3. Training Process

Data Preprocessing: The images are preprocessed by resizing and normalizing to match the input requirements of the Vision Transformer.

Feature Extraction: The ViTFeatureExtractor transforms images into feature vectors, capturing visual details that inform the generated poem.

Tokenization: The poetry dataset is tokenized using AutoTokenizer to convert captions into sequences that GPT-2 can process effectively, ensuring alignment with the model’s vocabulary.

Model Training: The model is trained using Seq2SeqTrainer from Hugging Face. This manages the integration between the Vision Transformer’s visual inputs and GPT-2’s text-based outputs, generating poems based on the images. The training process dynamically adjusts weights to improve performance over multiple epochs, with real-time evaluation.

## Usage
### Poetry Generation from Text Description

To generate poetry from a given image, preprocess the image using the ViTFeatureExtractor and feed it into the model, which will return a poetic description. Customize the number of words for the output.

### Training the Model

You can retrain the model by loading the datasets, preprocessing the images, and running the training script with Seq2SeqTrainer and Seq2SeqTrainingArguments.


