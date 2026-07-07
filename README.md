# 🏎️ AI Car Price Valuation Engine

An end-to-end Machine Learning web application that predicts the market value of used vehicles using a customized regression pipeline. The project features a fully engineered data pipeline, an optimized tree-based regression model, and a clean web-based user interface for live deployments.

---

## 🚀 Key Features

* **End-to-End Pipeline:** Seamlessly connects a raw dataset to a real-time predictive web interface.
* **Feature Engineering:** Implements programmatic column cleaning and automatic One-Hot Encoding to convert categorical data into clean mathematical indicators.
* **Hyperparameter Tuning:** Optimized tree constraints (`max_depth`) to minimize error metrics and maximize pattern recognition accuracy.
* **Full-Stack Deployment:** Built a responsive, lightweight localized web server using Flask, semantic HTML, and custom modern CSS variables.

---

## 📊 Model Performance Metrics

The model was evaluated using standard regression metrics on an unseen testing split (20% of total data):

* **Mean Absolute Error (MAE):** `174,196.75` 
  * *Interpretation:* On average, the model's price prediction is off by roughly 1.74 lakh currency units, making it highly effective for regional platform datasets.
* **R-squared ($R^2$) Score:** `0.4400` (44.0%)
  * *Interpretation:* The model captures 44.0% of the variance in pricing trends using only foundational vehicle specifications.

---

## 🛠️ Tech Stack & Architecture

* **Backend / Core ML:** Python, Pandas, NumPy, Scikit-Learn
* **Web Deployment:** Flask (Python Micro-framework)
* **Frontend UI:** HTML5, CSS3 (Custom Dark-Theme Variables)
* **Model Serialization:** Pickle (Binary protocol for saving trained model state)

---

## 📁 Repository Structure

```text
Real ML project/
│
├── main.py              # Model training, data preprocessing, and evaluation script
├── app.py               # Flask backend deployment controller 
├── car_data.csv        # Source vehicle dataset containing market values
├── car_model.pkl        # Serialized, trained Decision Tree Regressor model
└── templates/
    └── index.html       # Web application frontend user interface