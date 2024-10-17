import streamlit as st
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, date, time, timedelta
from function import *
import random
from joblib import load
import pickle

def prediction_user_input():
    st.sidebar.header("Input Transaction Data")

    # Initialize session state for the DataFrame if it doesn't exist
    if 'transactions' not in st.session_state:
        st.session_state.transactions = pd.DataFrame(columns=[
            'trb102', 'Transaction_Date', 'Previous_Transaction_Date', 
            'rtomod', 'trb004', 'merchant_description'
        ])

    # Input fields
    transaction_date = st.sidebar.date_input("Transaction Date", date.today())
    transaction_time = st.sidebar.slider("Transaction Time", time(1, 0))  # Default time set to 12:00 PM
    previous_transaction_date = st.sidebar.date_input("Previous Transaction Date", date.today())
    previous_transaction_time = st.sidebar.slider("Previous Transaction Time", time(0, 0))  # Default time set to 11:00 AM

    # Combine date and time into datetime
    transaction_datetime = pd.to_datetime(datetime.combine(transaction_date, transaction_time))
    previous_transaction_datetime = pd.to_datetime(datetime.combine(previous_transaction_date, previous_transaction_time))

    # Options for the select boxes
    rtomod_options = ["igate", "atm2", "atm", "node-manager", "eb", "teller"]
    rtomod = st.sidebar.selectbox("rtomod", rtomod_options)

    # Number input for amount and frequency
    amount = st.sidebar.slider("Transaction Amount", min_value=10000, max_value=10000000, step=10000)
    merchant_options = ['Mobile Banking', 'ATM', 'Deposit Machine', 'POS', 'Internet Banking', 'Teller', 'Kiosk']
    merchant_description = st.sidebar.selectbox("Merchant_Description", merchant_options)

    random_number = random.randint(10000, 99999)

    # Create a new row of data
    new_row = {
        'trb102': '99999',
        'trkey':f'{random_number}',
        "Transaction_Date": transaction_datetime,
        "Previous_Transaction_Date": previous_transaction_datetime,
        "rtomod": rtomod,
        "trb004": amount,
        "merchant_description": merchant_description,
    }

    # Add new row to the session state DataFrame when button is clicked
    if st.sidebar.button("Add Transaction"):
        # Create a DataFrame from the new row
        new_row_df = pd.DataFrame([new_row])
        
        # Concatenate the new row to the existing DataFrame
        st.session_state.transactions = pd.concat([st.session_state.transactions, new_row_df], ignore_index=True)

    # Show the DataFrame with all transactions
    st.write("### All Transactions")
    st.write(st.session_state.transactions)

    # Process the transactions DataFrame
    if not st.session_state.transactions.empty:
        df = st.session_state.transactions.copy()  # Work on a copy to prevent modifying the original
        
        # Perform calculations as before
        df = convert_date(df, 'Transaction_Date')
        df = calculate_frequencies(df, 'trb102','trkey', 'rtomod')
        df = previous_transaction(df, 'Transaction_Date', 'Previous_Transaction_Date')
        df = recency_score(df, 'time_diff_days')
        df = hour_stats(df)
        df = calculate_hour_bound(df, 'hour_mean', 'hour_std')
        df = calculate_hour_flag(df, 'hours', 'lower_hour_bound', 'upper_hour_bound')
        df = calculate_total_transaction_amount(df, 'trb102', 'trb004')
        df = calculate_mad(df, 'trb102', 'trb004')
        df = calculate_z_score(df, 'trb102')
        df = calculate_outlier_flags(df, 'trb102', 'trb004')
        df = calculate_total_transaction_amount_with_channel_days(df, 'trb102', 'rtomod', 'trb004', 'total_transaction_by_channel')

        st.write("### Processed Transaction Data")
        st.write(df)

        # Create a button for prediction
        if st.button("Predict"):

            # model = load('bank_ganesha_ai_behavior.joblib')
            file_path = 'model.pkl'
            with open(file_path , 'rb') as f:
                dict1 = pickle.load(f)

            features = [
            'rtomod',
            'trb004',
            'merchant_description',
            'frequency',
            'frequency_by_channel',
            'frequency_ratio',
            'recency_score',
            'lower_hour_bound',
            'upper_hour_bound',
            'hour_flag',
            'total_transaction_amount',
            'total_transaction_by_channel',
            'z_score',
            'amount_log_flag',
            'lower_amount_bound',
            'upper_amount_bound',
            'transaction_amount_flag'
            ]

            df_final = df[features]
            df_final = df_final.rename(columns={'trb004':'amount'})

            predictions = dict1.predict(df_final)

            df['Predictions'] = predictions
            
            st.write("### Prediction Results")
            st.write(df[['trb102', 'trkey', 'Predictions']]) 

if __name__ == '__main__':
    prediction_user_input()