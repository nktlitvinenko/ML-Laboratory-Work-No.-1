from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import onnxruntime as ort
from PIL import Image
import numpy as np
import io
from torchvision.datasets import CIFAR100

app = FastAPI()

class_names = CIFAR100(root="./data", train=False, download=True).classes
mean = np.array([0.5071, 0.4867, 0.4408])
std = np.array([0.2675, 0.2565, 0.2761])

session = ort.InferenceSession("resnet18_pretrained.onnx", providers=["CPUExecutionProvider"])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = image.resize((32, 32))
    image = np.asarray(image, dtype=np.float32) / 255.0
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0).astype(np.float32)

    outputs = session.run(None, {"input": image})
    logits = outputs[0]
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)

    predicted_class_idx = int(np.argmax(probabilities))
    predicted_class_name = class_names[predicted_class_idx]
    confidence = float(probabilities[0][predicted_class_idx]) * 100

    return JSONResponse(content={
        "predicted_class_index": predicted_class_idx,
        "predicted_class_name": predicted_class_name,
        "confidence_percent": round(confidence, 2)
    })