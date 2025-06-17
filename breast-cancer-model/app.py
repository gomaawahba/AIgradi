import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from PIL import Image
from flask import Flask, request, jsonify
import io

app = Flask(__name__)

# ----------- إعداد الجهاز -----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------- تحميل النموذج -----------
weights = EfficientNet_B3_Weights.DEFAULT
model = efficientnet_b3(weights=weights)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 7)  # عدد الفئات في FER2013

model.load_state_dict(torch.load("breast_cancer_model.pth", map_location=device))
model.to(device)
model.eval()

# ----------- التحويلات -----------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# ----------- دالة التنبؤ -----------
def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]

# ----------- endpoint التنبؤ -----------
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    try:
        img_bytes = file.read()
        prediction = predict_image(img_bytes)
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------- تشغيل السيرفر المحلي -----------
if __name__ == "__main__":
    app.run(debug=True)
