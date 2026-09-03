# MNIST Digit Prediction Model

An end-to-end Machine Learning project that classifies handwritten digits (0-9) using a trained Convolutional Neural Network (CNN) based on the **MNIST dataset**. This repository includes a web interface deployment pipeline engineered to host seamlessly on **Hugging Face Spaces**.

---

## 🚀 Live Demo
You can view and interact with the deployed model on Hugging Face Spaces here:
👉 **[Hugging Face Space Live App](https://huggingface.co/spaces/Dragonwarrior38/cnn_model)**

---

## 🛠️ Project Structure
```text
MNIST/
├── app/
│   └── upload.py        # Web application deployment & prediction logic
├── app.py               # Root application entry point for Hugging Face
├── requirements.txt     # Python package dependencies
├── README.md            # Project documentation
└── .gitignore           # Ignored system & model cache files
```

---

## 💻 Local Setup & Installation

Follow these steps to run the web application on your local machine:

### 1. Clone the Repository
```bash
git clone https://github.com
cd MNIST-prediction-model
```

### 2. Set Up a Virtual Environment (Recommended)
```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python app.py
```

---

## ☁️ Hugging Face Spaces Deployment

This repository is optimized to deploy directly into a Hugging Face Space application. 

### Environment Configuration
To keep authentication safe and satisfy GitHub Push Protection protocols, do **not** hardcode API tokens directly in `app/upload.py`. Instead, configure your Hugging Face credentials inside your script dynamically using environment variables:

```python
import os
# Fetch token safely from the host environment variable
hf_token = os.getenv("HF_TOKEN") 
```

Before building your space on Hugging Face, remember to go to your **Space Settings -> Repository Secrets**, and add your user access token using `HF_TOKEN` as the key name.

---

## 📦 Dependencies
The primary python frameworks used to build this pipeline include:
* **TensorFlow / Keras** - For running deep learning inference on the digit model
* **Gradio / Streamlit** - For building the interactive front-end web portal
* **NumPy & Pillow** - For image pre-processing and matrices normalization
