import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template
from torchvision import transforms
from PIL import Image

# Flask app setup
print("Starting Flask app...")
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'


# Vision Transformer Model Definition 
import torch
import torch.nn as nn
from torchvision import transforms
import timm

# Define your ViT wrapper
class ViTModel(nn.Module):
    def __init__(self, num_classes):
        super(ViTModel, self).__init__()
        # Load ViT model without pretraining
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=False)
        
        # Custom classification head
        self.vit.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.vit.head.in_features, num_classes)
        )

    def forward(self, x):
        return self.vit(x)

# Class labels
class_labels = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
num_classes = len(class_labels)

# Initialize model
model = ViTModel(num_classes=num_classes)

# === FIX: Load weights by prefixing keys with "vit." ===
# Load raw state dict
import requests

model_path = "model/bloodgroup.pt"

# If model doesn't exist, download it
if not os.path.exists(model_path):
    #print("Downloading model from external source...")
    
    # Replace this with a direct download link (NOT share link)
    url = "https://drive.google.com/uc?export=download&id=1Fl4w20c9IVZFy89yY71uPPKT8amGjO1Q"
    
    response = requests.get(url, stream=True)
    with open(model_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    #print("Model downloaded successfully.")

# Now load the model as usual
state_dict = torch.load(model_path, map_location=torch.device('cpu'))

# Fix key names: add "vit." prefix
new_state_dict = {}
for k, v in state_dict.items():
    if not k.startswith("vit."):
        new_state_dict["vit." + k] = v
    else:
        new_state_dict[k] = v

# Load updated state dict into model
model.load_state_dict(new_state_dict)
model.eval()

# Image transform for inference
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert 1-channel to 3-channel
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})

    file = request.files['file']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        img = Image.open(filepath).convert('RGB')  # Ensure compatibility
        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                top_idx = torch.argmax(probs).item()
                result = class_labels[top_idx]

        return jsonify({
                'success': True,
                'predicted_blood_group': result
            })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
