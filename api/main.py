import pandas as pd
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize the FastAPI app
app = FastAPI()

# --- Load Model and Data ---
mlflow.set_tracking_uri("http://127.0.0.1:5000")
RUN_ID = "671601d8c7674c188193c47eb56462e4"
MODEL_URI = f"runs:/{RUN_ID}/model"
DB_FILE = 'data/model_input.csv'

# --- Global Variables ---
model = None
df_sorted_results = pd.DataFrame() # This will hold our pre-calculated list

# --- Startup Event ---
@app.on_event("startup")
def load_and_predict():
    """
    This function runs ONCE when the uvicorn server starts.
    It loads the model and pre-calculates all churn probabilities.
    """
    global model, df_sorted_results
    
    # 1. Load Model
    try:
        model = mlflow.sklearn.load_model(MODEL_URI)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Load Data
    try:
        df_full_data = pd.read_csv(DB_FILE)
        print("Data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: {DB_FILE} not found.")
        return

    # 3. Pre-calculate Predictions (The "heavy lifting")
    print("Pre-calculating all user probabilities...")
    try:
        user_ids = df_full_data['user_id']
        X_to_predict = df_full_data.drop(['user_id', 'churn'], axis=1)
        
        churn_probabilities = model.predict_proba(X_to_predict)[:, 1]

        # Create the results DataFrame
        df_results = pd.DataFrame({
            'user_id': user_ids,
            'churn_probability': churn_probabilities
        })
        
        # Sort it *once* and save to our global variable
        df_sorted_results = df_results.sort_values(by='churn_probability', ascending=False)
        print("All probabilities calculated and sorted. API is ready.")
        
    except Exception as e:
        print(f"Error during pre-calculation: {e}")


# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"status": "Proactive Churn API is running."}


@app.get("/get-retention-list/")
def get_retention_list(top_n: int = 500):  # <-- !! HERE IS THE FIX !!
    """
    This is now super fast! It just slices the pre-calculated DataFrame.
    """
    if df_sorted_results.empty:
        return {"error": "Server is still initializing or failed to load data. Check API logs."}

    # Just take the top N rows from the list we already made
    df_top_n = df_sorted_results.head(top_n)
    
    # Convert to a dictionary (JSON-friendly) and return it
    return df_top_n.to_dict(orient='records')