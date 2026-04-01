from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # This allows your HTML file to talk to the API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model (make sure the filename matches exactly)
model = tf.keras.models.load_model('plant_model.h5')
class_names = ['Early_Blight', 'Late_Blight', 'Healthy']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read())).convert("RGB").resize((256, 256))
    # Note: We don't divide by 255 because your model has a rescaling layer
    img_batch = np.expand_dims(np.array(image), 0)
    
    predictions = model.predict(img_batch)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {"class": predicted_class, "confidence": float(confidence * 100)}
