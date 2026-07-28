import streamlit as st
import pickle
import json
import numpy as np
import base64


st.set_page_config(
    page_title="Bengaluru House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

def get_base64(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode()

image_path = r"C:\Users\hp\Desktop\bengaluru\Bangalore.jpg"
img = get_base64(image_path)

st.markdown(f"""
<style>

/* Don't blur the app */
.stApp {{
    background: transparent;
}}

/* Background image only */
.stApp::before {{
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;

    background: url("data:image/jpeg;base64,{img}") no-repeat center center;
    background-size: cover;

    filter: blur(4px) brightness(70%);
    transform: scale(1.08);

    z-index: -2;
}}

/* White cloudy overlay */
.stApp::after {{
    content: "";
    position: fixed;
    inset: 0;

    background: rgba(255,255,255,0.35);

    z-index: -1;
}}

</style>
""", unsafe_allow_html=True)



# Load model
model = pickle.load(open('house_price_model.pkl', 'rb'))

# Load columns
with open('columns.json', 'r') as f:
    data_columns = json.load(f)['data_columns']

# Location columns start after first 4 columns
locations = data_columns[4:]



st.markdown("""
<div style="
    background: linear-gradient(90deg, #1565C0, #1E88E5);
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    color: white;
    margin-bottom: 30px;
">

<h1 style="margin-bottom:10px;">
🏠 Bengaluru House Price Predictor
</h1>

<p style="font-size:18px; margin:0;">
Estimate the market value of your property instantly using Machine Learning.
</p>

</div>
""", unsafe_allow_html=True)

# Inputs
col1, col2 = st.columns(2)

with col1:
    sqft = st.number_input(
        "Total Square Feet",
        min_value=300,
        max_value=10000,
        value=1200
    )

    bhk = st.number_input(
        "BHK",
        min_value=1,
        max_value=10,
        value=2
    )

with col2:
    bath = st.number_input(
        "Bathrooms",
        min_value=1,
        max_value=10,
        value=2
    )

    balcony = st.number_input(
        "Balconies",
        min_value=0,
        max_value=10,
        value=1
    )

location = st.selectbox(
    "Location",
    locations
)


# Prediction
if st.button("Predict Price"):

    x = np.zeros(len(data_columns))

    x[0] = sqft
    x[1] = bath
    x[2] = balcony
    x[3] = bhk

    if location in data_columns:
        loc_index = data_columns.index(location)
        x[loc_index] = 1

    predicted_price = model.predict([x])[0]

    st.success(
        f"🏡 Estimated Price: ₹ {predicted_price:.2f} Lakhs"
    )

    price_per_sqft = (predicted_price * 100000) / sqft

    st.info(
        f"Approximate Price per Sqft: ₹ {price_per_sqft:,.0f}"
    )

    st.subheader("Property Summary")

    st.write(f"📍 Location: {location}")
    st.write(f"📐 Area: {sqft} sqft")
    st.write(f"🛏️ BHK: {bhk}")
    st.write(f"🚿 Bathrooms: {bath}")
    st.write(f"🌇 Balconies: {balcony}")
    st.markdown("---")

st.markdown("""
<div style="
    background-color:#1565C0;
    color:white;
    padding:20px;
    border-radius:12px;
    margin-top:20px;
">
    <h3 style="margin-top:0;">About this Application</h3>
    <p style="font-size:16px; line-height:1.6;">
        This application estimates Bengaluru property prices using a
        machine learning model trained on over <b>12,000 housing records</b>.
        Enter the property details above to receive an estimated market price.
    </p>
</div>
""", unsafe_allow_html=True)