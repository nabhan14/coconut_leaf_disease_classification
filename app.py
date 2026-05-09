import os
import json
import torch
import torch.nn as nn
from flask import Flask, render_template, request
from torchvision import transforms, models
from PIL import Image

# =========================
# FLASK APP
# =========================
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD CLASS NAMES
# =========================
with open("models/class_names.json", "r") as f:
    class_names = json.load(f)

# =========================
# LOAD MODEL
# =========================
model = models.resnet18(weights=None)

num_features = model.fc.in_features

model.fc = nn.Linear(num_features, len(class_names))

model.load_state_dict(
    torch.load("models/best_model.pth", map_location=device)
)

model = model.to(device)

model.eval()

# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================
# HOME PAGE
# =========================
@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":

        file = request.files["image"]

        if file:

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                file.filename
            )

            file.save(filepath)

            image = Image.open(filepath).convert("RGB")

            image = transform(image)

            image = image.unsqueeze(0)

            image = image.to(device)

            with torch.no_grad():

                outputs = model(image)

                probabilities = torch.nn.functional.softmax(
                    outputs[0],
                    dim=0
                )

                confidence_value, predicted = torch.max(
                    probabilities,
                    0
                )

            prediction = class_names[predicted.item()]

            confidence = round(
                confidence_value.item() * 100,
                2
            )

            image_path = '/' + filepath

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path
    )

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)