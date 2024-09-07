from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import cv2

app = Flask(__name__)

# 모델 경로 설정
bert_model_path = 'model/bert_scam_classifier'
autoencoder_model_path = 'model/autoencoder_model.keras'

# BERT 모델과 토크나이저 로드
bert_model = BertForSequenceClassification.from_pretrained(bert_model_path)
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_path)

# Autoencoder 모델 로드
autoencoder = tf.keras.models.load_model(autoencoder_model_path)

def predict(text):
    # 텍스트를 토큰화하고 모델을 통해 예측을 수행
    inputs = bert_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return prediction

def calculate_psnr(original, reconstructed):
    # 원본 이미지와 복원된 이미지의 PSNR 값을 계산
    original = (original[0] * 255.0).astype(np.uint8)
    reconstructed = (reconstructed[0] * 255.0).astype(np.uint8)
    
    if original.shape != reconstructed.shape:
        raise ValueError("Original and reconstructed images have different shapes")
    
    psnr_value = cv2.PSNR(original, reconstructed)
    return psnr_value

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
    
    try:
        # 원본 이미지와 복원된 이미지의 차원 변환 (예시: 4D 배열을 3D 배열로 변환)
        img_array = img_array.squeeze()  # (1, height, width, channels) -> (height, width, channels)
        reconstructed_img = reconstructed_img.squeeze()
        
        psnr_value = calculate_psnr(img_array, reconstructed_img)
    except ValueError as e:
        return jsonify({"error": str(e)}), 500

    if psnr_value > 18:
        return jsonify({"message": "The image is likely a genital image.", "psnr": psnr_value})
    else:
        return jsonify({"message": "The image is not a genital image.", "psnr": psnr_value})

# Hello World 라우트 추가
@app.route('/hello', methods=['GET'])
def hello_world():
    return jsonify({'message': 'Hello World!'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
