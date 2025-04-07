---
title: "Diabetes Prediction App"
emoji: "🍩"
colorFrom: "green"   # تغيير اللون إلى الأخضر
colorTo: "green"     # تغيير اللون إلى الأخضر
sdk: "streamlit"
sdk_version: "1.0.0"
app_file: "app.py"
pinned: true
---

# Diabetes Prediction App

This is a simple web application built using **Streamlit** that predicts whether a person is likely to have diabetes based on medical inputs such as glucose level, insulin, BMI, and more.

## Features
- Easy-to-use web interface
- Input fields for relevant health metrics
- Uses a trained machine learning model (Random Forest or any other classifier)
- Gives a binary prediction: Diabetic / Not Diabetic

## How It Works
The model was trained on the [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database). Once the user fills in the required data and clicks the **Predict** button, the model provides a prediction in real time.

## Tech Stack
- Python
- Streamlit
- Scikit-learn
- NumPy
- Hugging Face Spaces (for hosting)

## Try it now
Simply click the **"Open in Spaces"** button above and try the app directly in your browser – no installation needed!

## Files
- `app.py`: The main Streamlit app
- `diabetes_model.pkl`: The pre-trained model
- `requirements.txt`: Python dependencies
- `README.md`: This file

---

## Author
Huda Maher  
[LinkedIn](https://www.linkedin.com/in/huda-maher) | [GitHub](https://github.com/HADAHENO)

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

