import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# STAGE 1: LOAD & CLEAN DATA
# ==========================================
print("--- Stage 1: Loading Dataset ---")

# In a real setup, download a vehicle dataset from Kaggle and name it 'car_data.csv'
# For testing right now, we will create a quick 5-row sample so your code runs instantly!
try:
    df = pd.read_csv('car_data.csv')
except FileNotFoundError:
    print("car_data.csv not found! Creating a temporary sample dataset...")
    sample_data = {
        'Year': [2014, 2016, 2017, 2015, 2018],
        'Driven_Kms': [145000, 70000, 50000, 90000, 35000],
        'Fuel_Type': ['Petrol', 'Petrol', 'Diesel', 'Diesel', 'Petrol'],
        'Transmission': ['Manual', 'Automatic', 'Manual', 'Manual', 'Automatic'],
        'Selling_Price': [4.5, 18.2, 22.0, 12.5, 28.0] # Target in Lakhs or thousands
    }
    df = pd.DataFrame(sample_data)
    df.to_csv('car_data.csv', index=False)

# Display initial overview
# print(df.head())
# print(f"\nTotal rows loaded: {len(df)}")
# print(f"Missing values per column:\n{df.isnull().sum()}")

# ==========================================
# STAGE 2: FEATURE ENGINEERING (ONE-HOT ENCODING)
# ==========================================
print("\n--- Stage 2: Encoding Categorical Data ---")

# Convert text categories (Fuel, Transmission) into 1s and 0s
df_encoded = pd.get_dummies(df, columns=['fuel', 'transmission', 'seller_type', 'owner'], drop_first=True)

# print("Encoded DataFrame columns:")
# print(df_encoded.columns.tolist())

# --- 2. Define Features (X) and Target (y) ---
# We want to use Year, Kilometers Driven, and our new encoded fuel/transmission checkboxes
# features = ['year', 'km_driven', 'fuel_Diesel', 'fuel_Electric', 'fuel_LPG', 'fuel_Petrol', 'transmission_Manual']

# --- 3. Update Features (X) and Target (y) ---
# We exclude 'name' (since every car name is unique text) and 'selling_price' (our target)
X = df_encoded.drop(columns=['name', 'selling_price'])
y = df_encoded['selling_price'] # The price we want to predict

# --- 3. Split into 80% Training and 20% Testing ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Initialize and Train the Regressor Model ---
print("\n--- Stage 3: Training the Model ---")
model = DecisionTreeRegressor(max_depth=8, random_state=42)
model.fit(X_train, y_train)

# --- 5. Test the Model ---
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print("Model Training Complete!")
# print(f"Mean Absolute Error (MAE): {mae:.2f} (in your price currency unit)")
print(f"Updated Mean Absolute Error (MAE): {mae:.2f}")


# Calculate the R2 Score comparing true values with your predictions
r2 = r2_score(y_test, predictions)

print(f"R-squared (R2) Score: {r2:.4f} ({r2*100:.1f}% accurate patterns found)")


# ==========================================
# STAGE 5: LIVE PREDICTION FOR MARUTI ALTO (2007)
# ==========================================
print("\n--- Stage 5: Predicting Price for Maruti Alto 2007 ---")

# 1. Create a dictionary template filled with 0s matching your features
maruti_car_data = {col: 0 for col in X.columns}

# 2. Extract values directly from your comma-separated row:
maruti_car_data['year'] = 2007                 # 2007 model
maruti_car_data['km_driven'] = 125000          # 125,000 kilometers driven
maruti_car_data['fuel_Petrol'] = 1             # Petrol car
maruti_car_data['transmission_Manual'] = 1     # Manual transmission
maruti_car_data['seller_type_Individual'] = 1  # Sold by an Individual

# Note: 'owner_Second Owner' etc. remain 0 because it's a "First Owner" car!

# 3. Format into a DataFrame matching the model's exact expected shape
maruti_df = pd.DataFrame([maruti_car_data])

# 4. Run the data through your trained tree
maruti_predicted_price = model.predict(maruti_df)

print("\nVehicle Analysis Summary:")
print(f"-> Target Car: 2007 Maruti Alto LX (First Owner, Manual, 125k km)")
print(f"💰 AI Estimated Market Price: {maruti_predicted_price[0]:,.2f} (Lakhs/Units)")


# Save the trained model to a file named 'car_model.pkl'
with open('car_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("\n💾 Success! 'car_model.pkl' has been saved to your project folder.")