# Customer Choas Prediction
This is a **Streamlit web app** for predicting customer churn using a **CatBoost model**. Users can either input customer details manually or upload a CSV file to predict churn for multiple customers.
 
---
 
## 🛠️ Files
 
- `app.py` : Streamlit application code.  
- `catboost_best_model.pkl` : Trained CatBoost model.  
- `preprocessing_tools.pkl` : Preprocessing pipeline (scaling + encoding).  
- `TelcoChurn_Preprocessed.csv` : Dataset before scaling/encoding.  
- `TelcoChurn_Processed.csv` : Dataset after scaling/encoding.  
- `requirements.txt` : Python dependencies.  
- `.gitignore` : Files and folders to ignore in Git.
 
---
 
## 🚀 Deployment on Streamlit Cloud
 
1. Fork or clone this repository.  
2. Ensure all `.pkl` and CSV files are included.  
3. Go to [Streamlit Cloud](https://streamlit.io/cloud).  
4. Click **New App**, connect your GitHub repository, and select `app.py`.  
5. Streamlit will automatically install dependencies from `requirements.txt`.  
 
---
 
## 💡 Features
 
- Predict churn probability for a single customer.  
- Display predicted result along with probability.  
- Optional: upload CSV for batch predictions (to be added in next iteration).  
 
---
 
## ⚙️ Usage
 
1. Open the deployed app.  
2. Fill in customer details in the form.  
3. Click **Predict Churn**.  
4. View the predicted result and probability.  
 
---
 
## 📄 License
 
This project is licensed under the MIT License.
 
