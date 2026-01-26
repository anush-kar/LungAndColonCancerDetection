# Lung and Colon Cancer Detection Web App

A Flask-based deep learning web application for detecting **Lung Cancer**, **Colon Cancer**, or **Both** from CT scan images.  
This project implements a CNN-based medical image classification pipeline and is based on the research paper included in this repository.

---

## 📌 Project Overview

This application allows users to:
- Select a cancer detection type (Lung / Colon / Both)
- Upload a CT scan image
- Run a trained deep learning model
- View the prediction result directly in the browser

The system is designed for educational and research purposes and demonstrates the practical implementation of deep learning in medical image analysis.

---

## 🧠 Application Flow

1. User opens the web application.
2. User selects one of the three options:
   - **Lung Cancer**
   - **Colon Cancer**
   - **Both**
3. User uploads a CT scan image.
4. User clicks **Analyze**.
5. The trained deep learning model processes the image.
6. The prediction result is displayed on the screen.

---

## 🛠️ Tech Stack

- **Backend:** Flask (Python)
- **Deep Learning:** TensorFlow / Keras
- **Frontend:** HTML, CSS
- **Image Processing:** OpenCV, PIL
- **Model Type:** Convolutional Neural Network (CNN)

---

## 📂 Repository Structure

LungAndColonCancerDetection/\
│\
├── app.py\
├── requirements.txt/\
├── static/\
├── templates/\
├── models/\
├── research_paper.pdf\
└── README.md

---

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/anush-kar/LungAndColonCancerDetection.git
cd LungAndColonCancerDetection
```
### 2️⃣ Install Python & pip
Make sure Python 3.8+ and pip are installed.

### Check installation:

```bash
python --version
pip --version
```
### 3️⃣ Install Required Dependencies
```
pip install -r requirements.txt
```
### 4️⃣ Run the Flask Application
```
python app.py
```
### 5️⃣ Open in Browser
Once the server starts, open your browser and go to: http://127.0.0.1:5000/

## 🧪 How to Use the App
1️⃣ Open the app in your browser.

2️⃣ Choose one option:
- Lung Cancer
- Colon Cancer
- Both

3️⃣ Upload a CT scan image.

4️⃣ Click Analyze.

View the detection result displayed on the screen.

## 🖼️ Demo Screenshots
Add these images to a folder like screenshots/ in the repository and update paths if needed.

1️⃣ Lung Cancer Image Upload
![](screenshots\HF-lung-demo1.png)
2️⃣ Lung Cancer Detection Result
![](screenshots\HF-lung-demo2.png)

3️⃣ Colon Cancer Image Upload
![](screenshots\HF-colon-demo1.png)

4️⃣ Colon Cancer Detection Result
![](screenshots\HF-colon-demo2.png)

## 📄 Research Paper
This software implementation is based on the research paper included in the repository:

📘 Speech Emotion Recognition Using LSTM Architecture
(Adapted and extended for medical image classification use cases)

The paper explains the theoretical background, model architecture, dataset handling, and evaluation metrics used as the foundation for this project.

## ⚠️ Disclaimer
This project is intended only for educational and research purposes.
It is not a substitute for professional medical diagnosis.

## 👤 Author
Anush Kar \
GitHub: @anush-kar
