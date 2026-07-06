from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# 1. Load the trained model when the server starts
with open('car_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Get the exact feature names the model expects from its internal structure
MODEL_FEATURES = model.feature_names_in_.tolist()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 2. Extract inputs sent from the HTML form
        year = int(request.form['year'])
        km_driven = int(request.form['km_driven'])
        fuel = request.form['fuel']
        transmission = request.form['transmission']
        
        # 3. Create a dictionary template with 0 for every single expected feature
        custom_car_data = {col: 0 for col in MODEL_FEATURES}
        
        # Assign base numerical features
        if 'year' in custom_car_data:
            custom_car_data['year'] = year
        if 'km_driven' in custom_car_data:
            custom_car_data['km_driven'] = km_driven
        
        # 4. Turn on the exact checkboxes based on form drop-downs
        fuel_col = f'fuel_{fuel}'
        if fuel_col in custom_car_data:
            custom_car_data[fuel_col] = 1
            
        trans_col = f'transmission_{transmission}'
        if trans_col in custom_car_data:
            custom_car_data[trans_col] = 1

        # Default fallback for seller_type or owner if they exist in the dataset
        # (Since they aren't fields in our basic HTML form yet, they remain cleanly at 0)
            
        # 5. Format into a DataFrame matching the model's exact shape
        input_df = pd.DataFrame([custom_car_data])
        
        # Make the prediction
        predicted_price = model.predict(input_df)[0]
        formatted_price = f"{predicted_price:,.2f}"
        
        return render_template('index.html', prediction_text=f"AI Estimated Price: {formatted_price} units")
        
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error in prediction: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)