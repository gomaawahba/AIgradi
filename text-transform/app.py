from model_def import CustomTransformerModel
import pickle
# باقي الاستيرادات

with open("transformer_model_full.pkl", "rb") as f:
    data = pickle.load(f)
from model_def import CustomTransformerModel
import pickle
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

# تحميل البيانات
with open("transformer_model_full.pkl", "rb") as f:
    data = pickle.load(f)

model = data['model']
tokenizer = data['tokenizer']
label_encoder = data['label_encoder']
max_length = data['max_length']

model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data_json = request.json
    text = data_json.get('text', '')
    tokens = tokenizer.encode(text)
    input_tensor = torch.tensor(tokens.ids).view(1, -1)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        label = label_encoder.inverse_transform(predicted.cpu().numpy())[0]

    return jsonify({'prediction': label})

# if __name__ == "__main__":
#     app.run(debug=True)
if __name__ == "__main__":
    app.run(debug=True, port=3000)

