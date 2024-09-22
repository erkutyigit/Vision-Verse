import os
import datasets
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset
from transformers import (
    VisionEncoderDecoderModel,
    ViTFeatureExtractor,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)

# Device Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model and Tokenizer Loading
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
tokenizer = AutoTokenizer.from_pretrained('gpt2')


tokenizer.pad_token = tokenizer.eos_token



# Vision Encoder Decoder Model creating
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    'google/vit-base-patch16-224-in21k', 'gpt2'
)

model.to(device)  

# Config settings
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# Flickr30k dataset 
flickr_dataset = datasets.load_dataset("flickr_dataset_path", split="train")

# Gutenberg poem dataset 
poetry_df = pd.read_parquet('gutemberg_poetry_path')
poetry_texts = poetry_df['line'].tolist()


class VisionVerseDataset(Dataset):
    def __init__(self, image_dataset, poetry_texts, feature_extractor, tokenizer):
        self.image_dataset = image_dataset
        self.poetry_texts = poetry_texts
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        # Take the image and corresponding poem
        image = self.image_dataset[idx]['image']
        image = self.feature_extractor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        caption = self.poetry_texts[idx]
        inputs = self.tokenizer(caption, max_length=512, padding="max_length", truncation=True, return_tensors="pt")

        return {
            'pixel_values': image,
            'labels': inputs['input_ids'].squeeze(0)
        }


# Splitting dataset for training and testing
train_size = int(0.8 * len(flickr_dataset))
val_size = len(flickr_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(flickr_dataset, [train_size, val_size])

# Dataset objects
vision_verse_train_dataset = VisionVerseDataset(train_dataset, poetry_texts, feature_extractor, tokenizer)
vision_verse_val_dataset = VisionVerseDataset(val_dataset, poetry_texts, feature_extractor, tokenizer)

# DataLoader 
train_dataloader = torch.utils.data.DataLoader(vision_verse_train_dataset, batch_size=16, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(vision_verse_val_dataset, batch_size=16)

# Training parameter tuning
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    evaluation_strategy="steps",
    logging_steps=10,
    save_steps=10,
    eval_steps=10,
    warmup_steps=500,
    save_total_limit=2,
    num_train_epochs=3,
    fp16=torch.cuda.is_available()
)

# Using Seq2SeqTrainer starting the training
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=vision_verse_train_dataset,
    eval_dataset=vision_verse_val_dataset,
    data_collator=default_data_collator,
    tokenizer=tokenizer
)

# Model Training
trainer.train()


# Generating poem from image
def generate_poetry_from_image(image_path, model, tokenizer, feature_extractor, max_length=50):
    image = Image.open(image_path).convert("RGB")
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)

    generated_ids = model.generate(pixel_values, max_length=max_length, num_beams=4)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return generated_text


# Example usage
image_path = 'example_image.jpg'
poetry_output = generate_poetry_from_image(image_path, model, tokenizer, feature_extractor)
print(poetry_output)
