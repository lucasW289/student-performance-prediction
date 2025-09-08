# EduPredict: Student Performance Predictor

---

## Project Overview
EduPredict is an interactive machine learning app that predicts whether a student will **Pass** or **Fail** based on study hours, attendance, past exam scores, parental education, internet access, and extracurricular activities.

The project includes:  
- End-to-end data preprocessing and machine learning pipeline  
- RandomForest classifier with feature importance analysis  
- Real-time prediction through a **Streamlit** web app  

---
## Demo
Experience the live application here: [EduPredict Streamlit App](https://student-performance-prediction-g17.streamlit.app/)

---
## Project Structure

- `student_performance_dataset.csv` – Dataset of student information  
- `train_model.py` – Script for training the ML pipeline  
- `student_performance_pipeline.kpl` – Trained ML pipeline  
- `student_performance_original_features.kpl` – Original feature names  
- `student_performance_label_encoder.kpl` – Label encoder for Pass/Fail target  
- `student_performance_feature_importances.kpl` – Feature importance data  
- `app.py` – Streamlit app for predictions  

---

## Features

- Interactive input for student information  
- Preprocessing and encoding using StandardScaler and OneHotEncoder  
- RandomForest classifier with feature importance  
- Model evaluation with accuracy, classification report, and confusion matrix  

---

## Tools & Technologies

- **Frontend:** Streamlit, Matplotlib, Seaborn  
- **ML Model:** Scikit-learn (RandomForest, Pipeline, ColumnTransformer), Pandas, Numpy, Joblib  
- **Data Processing:** Label Encoding, OneHotEncoding, StandardScaler  

