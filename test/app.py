from flask import Flask, request, jsonify, send_file
from model import estimate_fish, preprocess_image
import os
import base64
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/estimate', methods=['POST'])
def estimate():
    # 檢查影像資料
    image_base64 = request.json.get('image_base64')
    if not image_base64:
        return jsonify({"error": "Missing image_base64"}), 400

    # 影像解碼並儲存
    image_data = base64.b64decode(image_base64)
    image_path = "latest_sonar.png"
    with open(image_path, "wb") as f:
        f.write(image_data)

    try:
        # 推論魚隻數量
        image_array = preprocess_image(image_path, target_size=(320, 576))
        estimated_count = int(estimate_fish(image_array))  # 回傳整數
        
        # 將影像轉回 base64 回傳
        with open(image_path, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode('utf-8')

        return jsonify({
            "estimated_count": estimated_count,
            "image_base64": encoded_image
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8800)