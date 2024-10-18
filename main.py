import streamlit as st
from prediction_user_input import prediction_user_input
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, date, time, timedelta
from function import *
import random
from joblib import load
from prediction_csv_input import predict_from_csv

def main():
    st.title("AI Behavior predict simulation")

    # Select prediction method
    prediction_method = st.sidebar.selectbox(
        "Choose Prediction Method",
        ("User Input", "CSV Upload")
    )

    if prediction_method == "User Input":
        st.write("### Predict from User Input")
        prediction_user_input()

    else:
        st.write("### Predict from CSV Upload")
        predict_from_csv()

if __name__ == "__main__":
    main()
