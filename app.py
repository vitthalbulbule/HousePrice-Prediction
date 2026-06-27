import streamlit as st
import pandas as pd
import pickle
import numpy as np

# =====================================
# Load Trained Model
# =====================================
with open("house_price_model.pkl", "rb") as f:
    model = pickle.load(f)

# =====================================
# Page Configuration
# =====================================
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Prediction")
st.write("Predict House Prices using Machine Learning")

# =====================================
# User Inputs
# =====================================
col1, col2 = st.columns(2)

with col1:

    overall_qual = st.slider(
        "Overall Quality",
        min_value=1,
        max_value=10,
        value=5
    )

    gr_liv_area = st.number_input(
        "Ground Living Area (sq ft)",
        min_value=300,
        max_value=10000,
        value=1500
    )

    garage_cars = st.slider(
        "Garage Capacity",
        min_value=0,
        max_value=5,
        value=2
    )

    garage_area = st.number_input(
        "Garage Area",
        min_value=0,
        max_value=2000,
        value=500
    )

    total_bsmt_sf = st.number_input(
        "Basement Area",
        min_value=0,
        max_value=5000,
        value=800
    )

    lot_area = st.number_input(
        "Lot Area",
        min_value=1000,
        max_value=250000,
        value=10000
    )

    year_built = st.number_input(
        "Year Built",
        min_value=1900,
        max_value=2025,
        value=2000
    )

with col2:

    full_bath = st.slider(
        "Full Bathrooms",
        min_value=0,
        max_value=6,
        value=2
    )

    fireplaces = st.slider(
        "Fireplaces",
        min_value=0,
        max_value=5,
        value=1
    )

    kitchen_qual = st.selectbox(
        "Kitchen Quality",
        [1, 2, 3, 4, 5]
    )

    exter_qual = st.selectbox(
        "Exterior Quality",
        [1, 2, 3, 4, 5]
    )

    neighborhood = st.text_input(
        "Neighborhood",
        value="NAmes"
    )

    garage_cond = st.selectbox(
        "Garage Condition",
        [1, 2, 3, 4, 5]
    )

    heating_qc = st.selectbox(
        "Heating Quality",
        [1, 2, 3, 4, 5]
    )

# =====================================
# Load Template Row
# =====================================
with open("template_row.pkl", "rb") as f:
    input_df = pickle.load(f)

# =====================================
# Update Features
# =====================================
input_df.loc[:, "OverallQual"] = overall_qual
input_df.loc[:, "GrLivArea"] = gr_liv_area
input_df.loc[:, "GarageCars"] = garage_cars
input_df.loc[:, "GarageArea"] = garage_area
input_df.loc[:, "TotalBsmtSF"] = total_bsmt_sf
input_df.loc[:, "LotArea"] = lot_area
input_df.loc[:, "YearBuilt"] = year_built
input_df.loc[:, "FullBath"] = full_bath
input_df.loc[:, "Fireplaces"] = fireplaces
input_df.loc[:, "KitchenQual"] = kitchen_qual
input_df.loc[:, "ExterQual"] = exter_qual
input_df.loc[:, "Neighborhood"] = neighborhood
input_df.loc[:, "GarageCond"] = garage_cond
input_df.loc[:, "HeatingQC"] = heating_qc

# =====================================
# Prediction
# =====================================
if st.button("Predict Price"):

    try:

        pred_log = model.predict(input_df)[0]

        predicted_price = np.expm1(pred_log)

        st.success(
            f"Estimated House Price: ₹ {predicted_price:,.0f}"
        )

        st.metric(
            label="Predicted House Price",
            value=f"₹ {predicted_price:,.0f}"
        )

        st.write("Log Prediction:", pred_log)

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# =====================================
# Debug Section
# =====================================
with st.expander("Show Input Data"):

    st.dataframe(input_df)

# =====================================
# Footer
# =====================================
st.markdown("---")
st.caption(
    "Built using Streamlit, Scikit-Learn, Linear Regression, Ordinal Encoding, One-Hot Encoding and Robust Scaling."
)