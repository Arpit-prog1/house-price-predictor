import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("🏠 House Price Predictor")
st.write("Enter the house details and get an estimated price.")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sq ft)", min_value=500, max_value=20000, value=5000, step=100)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=8, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=6, value=2)
    stories = st.number_input("Stories", min_value=1, max_value=5, value=2)
    parking = st.number_input("Parking spaces", min_value=0, max_value=5, value=1)
    furnishingstatus = st.selectbox(
        "Furnishing status", ["furnished", "semi-furnished", "unfurnished"]
    )

with col2:
    mainroad = st.selectbox("Main road access", ["yes", "no"])
    guestroom = st.selectbox("Guest room", ["no", "yes"])
    basement = st.selectbox("Basement", ["no", "yes"])
    hotwaterheating = st.selectbox("Hot water heating", ["no", "yes"])
    airconditioning = st.selectbox("Air conditioning", ["yes", "no"])
    prefarea = st.selectbox("Preferred area", ["yes", "no"])

if st.button("Predict Price", type="primary"):
    new_house = pd.DataFrame([{
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "parking": parking,
        "prefarea": prefarea,
        "furnishingstatus": furnishingstatus,
    }])
    price = model.predict(new_house)[0]
    st.success(f"Estimated price: ₹ {price:,.0f}")
    st.caption("Model: regression trained on 545 houses (R² ≈ 0.65). Treat this as a rough estimate.")
