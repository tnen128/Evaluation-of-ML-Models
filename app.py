from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Initialize Flask app
app = Flask(__name__)
model = joblib.load("logisticregression.pkl")
scaler = StandardScaler()

def prepare_features(data):
    # Convert categorical features to numerical format
    gender_map = {"Male": [0, 1], "Female": [1, 0]}  
    yes_no_map = {"Yes": 1, "No": 0}
    
    
    try:
       
        num_children_income_df = pd.DataFrame({
            'Num_Children': data['Num_Children'],
            'Income': data['Income']
        })
        
        # Fit and transform the scaler on new data
        scaled_values = scaler.fit_transform(num_children_income_df)
        
        Own_Car = [yes_no_map[car] for car in data['Own_Car']]
        Own_Housing = [yes_no_map[house] for house in data['Own_Housing']]
        
        # Split Gender into two columns: Gender_Female and Gender_Male
        Gender_Female = [gender_map[g][0] for g in data['Gender']]
        Gender_Male = [gender_map[g][1] for g in data['Gender']]
        
        # Combine all features into a numpy array
        features = np.column_stack([scaled_values, Own_Car, Own_Housing, Gender_Female, Gender_Male])
        return features
    
    except KeyError as e:
        raise ValueError(f"Missing field: {str(e)}")

# Define the prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse and validate input JSON data
        data = request.get_json(force=True)
        required_keys = ["Num_Children", "Gender", "Income", "Own_Car", "Own_Housing"]
        for key in required_keys:
            if key not in data:
                return jsonify({"error": f"Missing required field: {key}"}), 400
        
        # Ensure all input lists have the same length
        length = len(data['Num_Children'])
        for key in data:
            if len(data[key]) != length:
                return jsonify({"error": "All input lists must be of the same length."}), 400
        
        # Prepare features for prediction
        X = prepare_features(data)
        
        # Make predictions
        predictions = model.predict(X)
        return jsonify({"predictions": predictions.tolist()})
    
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
