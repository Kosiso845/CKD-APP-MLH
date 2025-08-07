# CKD-APP-MLH
Chronic Kidney Disease (CKD) Prediction App
A machine learning-powered web application that predicts the risk of Chronic Kidney Disease (CKD) using real clinical data. This app enables early detection using simple inputs like blood pressure, BMI, creatinine levels, and more.
 Live App: [Try the App Here](https://ckd-app-app-project-qmsb7lomjdzisxyiqkrkdb.streamlit.app/)
 GitHub Repo: github.com/Kosiso845/ckd-prediction-app
** What It Does**
Predicts CKD stages using a trained Random Forest Classifier (accuracy: 89%).
Accepts 10 basic health indicators from the user.
Displays prediction results clearly with options to download report.
Deployable and accessible via any browser (built with Streamlit).
**How We Built It**
Python (Pandas, NumPy, Scikit-learn) for data processing and model training.
Joblib to serialize the trained model.
Streamlit to build and deploy a lightweight, responsive user interface.
Dataset: UCI CKD Dataset (400 patient records).
Training script: ckd app.py
Deployment script: app.py
**ckd-prediction-app/**
│
├── app.py                  Streamlit frontend
├── ckd app.py         Model training script
├── tuned_rf_model.pkl      Saved Random Forest model
├── requirements.txt        Dependencies
├── ckd_data.csv           #Dataset 
└── README.md        
**Challenges We Ran Into**
Handling inconsistent missing values in the raw dataset.
Achieving a balance between model accuracy and interpretability.
Customizing the user interface for mobile responsiveness
** What We Learned**
Hands-on application of supervised machine learning in healthcare.
Feature engineering and data cleaning best practices.
Deploying ML apps with Streamlit.
Importance of clear UI/UX when building health-tech tools.
**Built With**
Python (Pandas, NumPy, Scikit-learn)
Streamlit
Joblib
Git & GitHub
UCI CKD Dataset

