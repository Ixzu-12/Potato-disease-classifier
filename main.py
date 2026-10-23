from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = tf.keras.models.load_model('plant_model.h5')
class_names = ['Early_Blight', 'Late_Blight', 'Healthy']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read())).convert("RGB").resize((256, 256))
   
    img_batch = np.expand_dims(np.array(image), 0)
    
    predictions = model.predict(img_batch)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {"class": predicted_class, "confidence": float(confidence * 100)}
