# Potato-disease-classifier
A deep learning model for predicting wether plant(potato) is healthy or not
# 🌿 Plant Disease Classifier API

A full-stack Machine Learning application that identifies plant diseases from leaf images using **Convolutional Neural Networks (CNN)** and serves predictions via a **FastAPI** web server.

## 🚀 Overview
This project automates the detection of plant pathologies, helping farmers and gardeners diagnose issues like **Early Blight** and **Late Blight** instantly. The model was trained on the PlantVillage dataset and achieves high accuracy in classifying potato and tomato leaf health.

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow, Keras
* **Backend:** FastAPI (Python)
* **Server:** Uvicorn
* **Image Processing:** Pillow (PIL), NumPy
* **Deployment:** Docker / Google Colab (via Localtunnel)

## 📂 Project Structure
* `main.py`: The FastAPI server containing the `/predict` endpoint.
* `plant_model.h5`: The trained TensorFlow model weights.
* `requirements.txt`: List of Python dependencies.
* `notebooks/`: The original Google Colab training script.

## 🚦 How to Run Locally
1. **Clone the repo:**
   ```bash
   git clone [https://github.com/your-username/plant-disease-classifier.git](https://github.com/your-username/plant-disease-classifier.git)
   cd plant-disease-classifier
