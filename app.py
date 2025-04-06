import streamlit as st 
import pandas as pd
import numpy as np 
import os
import pickle 
import warnings

st.set_page_config(page_title="Crop Recommender", page_icon="🌿", layout='centered', initial_sidebar_state="collapsed")

def load_model(modelfile):
    loaded_model = pickle.load(open(modelfile, 'rb'))
    return loaded_model

def main():
    st.markdown(
        """
        <div style="text-align:center;">
            <h1 style="color:MediumSeaGreen;">🌿 Crop Recommendation System 🌾</h1>
            <p style="font-size:18px;">Smart suggestions for what to grow based on your soil and climate 🌱</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

    with st.expander("ℹ️ About the App", expanded=False):
        st.write("""
        Crop recommendation is a key aspect of precision agriculture. This app uses machine learning to recommend the most suitable crops to grow based on soil nutrients and environmental conditions. It's simple — just fill in your farm's info and let the AI do the rest.
        """)

    with st.form("crop_form"):
        st.subheader("👨‍🌾 Enter your farm's data:")

        col1, col2 = st.columns(2)

        with col1:
            N = st.number_input("🌿 Nitrogen (N)", 0, 500)
            P = st.number_input("🌸 Phosphorus (P)", 0, 500)
            K = st.number_input("🍂 Potassium (K)", 0, 500)
            ph = st.number_input("🧪 Soil pH", 0.0, 14.0, step=0.1)

        with col2:
            temp = st.number_input("🌡️ Temperature (°C)", 0.0, 60.0, step=0.1)
            humidity = st.number_input("💧 Humidity (%)", 0.0, 100.0, step=0.1)
            rainfall = st.number_input("🌧️ Rainfall (mm)", 0.0, 500.0, step=0.1)

        submitted = st.form_submit_button("🚀 Predict Best Crop")

        if submitted:
            # Input validation check
            if N == 0 or P == 0 or K == 0 or temp == 0.0 or humidity == 0.0 or ph == 0.0 or rainfall == 0.0:
                st.error("❌ Please enter all values for an accurate prediction.")
            else:
                feature_list = [N, P, K, temp, humidity, ph, rainfall]
                single_pred = np.array(feature_list).reshape(1, -1)
                loaded_model = load_model('model.pkl')
                prediction = loaded_model.predict(single_pred)
                
                st.markdown("---")
                st.success(f"✅ Based on the inputs, the recommended crop is: **{prediction.item().title()}** 🌾")
                st.markdown("---")

    st.markdown(
        """
        <hr>
        <p style="font-size:14px; text-align:center;">
        ⚠️ This is a demo AI tool intended for educational purposes only.<br>
        📁 Source code available on <a href="https://github.com/Mjsadanand/AI-Powered-Sustainable-Farming" target="_blank">GitHub</a>.
        </p>
        """, 
        unsafe_allow_html=True
    )

    # Hide menu and footer
    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
