from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

# Load trained model and feature columns
model = joblib.load("model_pipeline.pkl")
feature_columns = joblib.load("feature_columns.pkl")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        try:
            # Collect form input
            form_data = {
                "area": float(request.form["area"]),
                "bedrooms": int(request.form["bedrooms"]),
                "bathrooms": int(request.form["bathrooms"]),
                "stories": int(request.form["stories"]),
                "parking": int(request.form["parking"]),
                "mainroad": 1 if request.form.get("mainroad") == "yes" else 0,
                "guestroom": 1 if request.form.get("guestroom") == "yes" else 0,
                "basement": 1 if request.form.get("basement") == "yes" else 0,
                "hotwaterheating": 1 if request.form.get("hotwaterheating") == "yes" else 0,
                "airconditioning": 1 if request.form.get("airconditioning") == "yes" else 0,
                "prefarea": 1 if request.form.get("prefarea") == "yes" else 0,
                "furnishingstatus_semi-furnished": 1 if request.form.get("furnishingstatus") == "semi-furnished" else 0,
                "furnishingstatus_unfurnished": 1 if request.form.get("furnishingstatus") == "unfurnished" else 0
            }

            # Derived features
            form_data["price_per_sqft"] = 0  # placeholder if used in training
            form_data["area_sqrt"] = np.sqrt(form_data["area"])
            form_data["area_per_bed"] = form_data["area"] / max(form_data["bedrooms"], 1)
            form_data["bath_per_bed"] = form_data["bathrooms"] / max(form_data["bedrooms"], 1)
            form_data["parking_per_bed"] = form_data["parking"] / (form_data["bedrooms"] + 1)
            form_data["amenity_count"] = sum([form_data[k] for k in ["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"]])
            form_data["area_pref"] = form_data["area"] * form_data["prefarea"]
            form_data["area_ac"] = form_data["area"] * form_data["airconditioning"]

            # Prepare input DataFrame
            input_df = pd.DataFrame([form_data], columns=feature_columns).fillna(0)

            # Predict
            log_price = model.predict(input_df)[0]
            price = np.exp(log_price)

            result = f"Estimated House Price: ₹{price:,.0f}"

        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
