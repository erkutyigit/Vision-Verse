import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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

# Cihaz ayarları
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model ve Tokenizer yükleme
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Padding tokenını ayarla (İki seçenekten birini kullan)
# 1. EOS token'ı padding olarak kullan
tokenizer.pad_token = tokenizer.eos_token

# 2. Veya yeni bir padding token ekle
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Vision Encoder Decoder Model oluşturma
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    'google/vit-base-patch16-224-in21k', 'gpt2'
)

model.to(device)  # Modeli GPU'ya yükleme

# Config ayarları
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# Flickr30k veri kümesini yükleme
flickr_dataset = datasets.load_dataset("C:/Users/Monster/PycharmProjects/AIE5601_TermProject/Images", split="train")

# Gutenberg şiir veri kümesini yükleme
poetry_df = pd.read_parquet('C:/Users/Monster/PycharmProjects/AIE5601_TermProject/gutemberg_poetry.parquet')
poetry_texts = poetry_df['line'].tolist()


# Resim ve metinlerin ön işlenmesi
class VisionVerseDataset(Dataset):
    def __init__(self, image_dataset, poetry_texts, feature_extractor, tokenizer):
        self.image_dataset = image_dataset
        self.poetry_texts = poetry_texts
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        # Görseli ve ona karşılık gelen şiiri al
        image = self.image_dataset[idx]['image']
        image = self.feature_extractor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        caption = self.poetry_texts[idx]
        inputs = self.tokenizer(caption, max_length=512, padding="max_length", truncation=True, return_tensors="pt")

        return {
            'pixel_values': image,
            'labels': inputs['input_ids'].squeeze(0)
        }


# Eğitim ve doğrulama için datasetlerin ayrılması
train_size = int(0.8 * len(flickr_dataset))
val_size = len(flickr_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(flickr_dataset, [train_size, val_size])

# Dataset objeleri oluşturma
vision_verse_train_dataset = VisionVerseDataset(train_dataset, poetry_texts, feature_extractor, tokenizer)
vision_verse_val_dataset = VisionVerseDataset(val_dataset, poetry_texts, feature_extractor, tokenizer)

# DataLoader oluşturma
train_dataloader = torch.utils.data.DataLoader(vision_verse_train_dataset, batch_size=16, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(vision_verse_val_dataset, batch_size=16)

# Eğitim parametrelerini ayarlama
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

# Seq2SeqTrainer kullanarak eğitimi başlatma
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=vision_verse_train_dataset,
    eval_dataset=vision_verse_val_dataset,
    data_collator=default_data_collator,
    tokenizer=tokenizer
)

# Modeli eğitme
trainer.train()


# Görsel girdiden şiir üretme fonksiyonu
def generate_poetry_from_image(image_path, model, tokenizer, feature_extractor, max_length=50):
    image = Image.open(image_path).convert("RGB")
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)

    generated_ids = model.generate(pixel_values, max_length=max_length, num_beams=4)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return generated_text


# Örnek kullanımı
image_path = 'example_image.jpg'
poetry_output = generate_poetry_from_image(image_path, model, tokenizer, feature_extractor)
print(poetry_output)
