from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# BERT 모델과 토크나이저 로드
bert_model_name = "gihakkk/bert_scam_classifier"
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertForSequenceClassification.from_pretrained(bert_model_name)

# Autoencoder 모델 다운로드 및 로드
autoencoder_model_url = "https://huggingface.co/gihakkk/autoencoder_model.keras/resolve/main/autoencoder_model.keras"
autoencoder_model_path = "autoencoder_model.keras"

# 모델 파일 다운로드
response = requests.get(autoencoder_model_url)
with open(autoencoder_model_path, "wb") as f:
    f.write(response.content)

# TensorFlow 모델 로드
autoencoder = tf.keras.models.load_model(autoencoder_model_path)

def predict(text):
    # 텍스트를 토큰화하고 모델을 통해 예측을 수행
    inputs = bert_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return prediction

def preprocess_image(image):
    # 이미지를 전처리하여 모델 입력 형식에 맞추기
    img = Image.open(io.BytesIO(image)).convert('RGB')
    img = img.resize((152, 152))  # 모델이 요구하는 크기
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    prediction = predict(text)
    return jsonify({'prediction': prediction})

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files['image'].read()
    img_array = preprocess_image(image)
    reconstructed_img = autoencoder.predict(img_array)
    
    # PSNR 계산 (직접 계산으로 대체)
    original = (img_array[0] * 255.0).astype(np.uint8)
    reconstructed = (reconstructed_img[0] * 255.0).astype(np.uint8)
    
    # PSNR 계산 함수 직접 구현
    def psnr(img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        return 20 * np.log10(max_pixel / np.sqrt(mse))
    
    psnr_value = psnr(original, reconstructed)

    if psnr_value > 18:
        return jsonify({"message": "The image is likely a genital image.", "psnr": psnr_value})
    else:
        return jsonify({"message": "The image is not a genital image.", "psnr": psnr_value})

@app.route('/hello', methods=['GET'])
def hello_world():
    return jsonify({'message': 'Hello World!'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
