import streamlit as st
import pandas as pd
import joblib
import pickle
from function import *  # Make sure to import all your necessary functions
import shap


def select_rows_by_index(df):
    # Initialize the session state to store selected row indices if not already done
    if 'selected_rows' not in st.session_state:
        st.session_state['selected_rows'] = []

    # Allow the user to add rows by index
    add_row_by_index = st.number_input(
        "Enter the row index to add:", 
        min_value=0, max_value=len(df) - 1, step=1
    )

    if st.button("Add Row by Index"):
        if add_row_by_index not in st.session_state['selected_rows']:
            st.session_state['selected_rows'].append(add_row_by_index)
            st.success(f"Row {add_row_by_index} added")
        else:
            st.warning(f"Row {add_row_by_index} is already added.")
    
    # Button to add all rows
    if st.button("Add All Rows"):
        st.session_state['selected_rows'] = list(df.index)
        st.success("All rows added")

    if len(st.session_state['selected_rows']) > 0:
        # Display the DataFrame with the selected rows
        selected_indices = st.session_state['selected_rows']
        df_selected = df.loc[selected_indices]
        st.write("Selected Rows for Prediction:")
        st.dataframe(df_selected)

        return df_selected
    else:
        st.warning("No rows have been selected yet.")
        return None


def predict_from_csv():
    # Load model
    # model = pickle.load('model.pkl')

    file_path = 'model.pkl'
    with open(file_path , 'rb') as f:
        dict1 = pickle.load(f)
    # File uploader
    # uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    # uploaded_file = pd.read_csv('cek.csv')

    # Automatically show the data once a CSV file is uploaded
    # if uploaded_file is not None:
        # Read and display the CSV data
    df = pd.read_csv('cek.csv')
    df['trb102'] = df['trb102'].astype(str)
    st.dataframe(df)

    df = select_rows_by_index(df)

    # Store the DataFrame in session state
    st.session_state['df'] = df

    # Show "Predict Data" button after data is loaded
    if 'df' in st.session_state:
        if st.button('Predict Data'):
            df = st.session_state['df']

            # Check if the necessary columns are present
            required_columns = [
                'trb102', 'rttime', 'rtomod', 'trb004', 'merchant_description'
            ]
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Missing required column: {col}")

            # Preprocessing steps
            df = convert_date(df, 'rttime')
            df = calculate_frequencies(df, 'trb102', 'trkey', 'rtomod')
            df = previous_transaction2(df, 'rttime', 'trb102', 'rtomod')
            df = recency_score(df, 'time_diff_days')
            df = hour_stats(df)
            df = calculate_hour_bound(df, 'hour_mean', 'hour_std')
            df = calculate_hour_flag(df, 'hours', 'lower_hour_bound', 'upper_hour_bound')
            df = calculate_total_transaction_amount(df, 'trb102', 'trb004')
            df = calculate_mad(df, 'trb102', 'trb004')
            df = calculate_z_score(df, 'trb102')
            df = calculate_outlier_flags(df, 'trb102', 'trb004')
            df = calculate_total_transaction_amount_with_channel_days(df, 'trb102', 'rtomod', 'trb004', 'total_transaction_by_channel')

            # Define the features for prediction
            features = [
                'rtomod', 'trb004', 'merchant_description', 'frequency',
                'frequency_by_channel', 'frequency_ratio', 'recency_score',
                'lower_hour_bound', 'upper_hour_bound', 'hour_flag',
                'total_transaction_amount', 'total_transaction_by_channel',
                'z_score', 'amount_log_flag', 'lower_amount_bound', 
                'upper_amount_bound', 'transaction_amount_flag'
            ]

            # Create a DataFrame for prediction features
            df_final = df[features]
            df_final = df_final.rename(columns={'trb004': 'amount'})  # Rename for the model


            # Make predictions
            predictions = dict1.predict(df_final)
           
            # Add predictions to the DataFrame
            df['Predictions'] = predictions

            # Display the predictions
            st.write("Data with Predictions:")
            st.dataframe(df[['trb102','rtdate','Predictions']])  # Show the predictions in the DataFrame

            model = dict1.named_steps['model']
            preprocessor = dict1.named_steps['preprocessor']
            transform_data = preprocessor.transform(df_final)
            feature_names = preprocessor.get_feature_names_out()
            # Generate SHAP values
            explainer = shap.TreeExplainer(model)  # Initialize the SHAP explainer with the model
            shap_values = explainer.shap_values(transform_data)  # Calculate SHAP values for the DataFrame


            shap_df = pd.DataFrame(shap_values, columns=feature_names)

            # Add the actual feature values to the DataFrame
            for feature in df_final.columns:
                shap_df[f"{feature}_value"] = df_final[feature].values

            # Function to get top N influential features for a row
            def get_top_features(row, n=3):
                feature_importances = [(feature, abs(value)) for feature, value in row.items() if not feature.endswith('_value') and feature not in ['prediction', 'true_label']]
                top_features = sorted(feature_importances, key=lambda x: x[1], reverse=True)[:n]
                return ', '.join([f"{feature} ({value:.4f})" for feature, value in top_features])

            # Add a column with top 5 influential features
            shap_df['top_3_features'] = shap_df.apply(get_top_features, axis=1)

            st.dataframe(shap_df['top_3_features'])

            # Add SHAP values to the DataFrame (for example, the sum of SHAP values per row)
            # df['SHAP_sum'] = shap_values.sum(axis=1)

            # Display SHAP values for each row in the DataFrame
            # st.write("Data with SHAP Values:")
            # st.dataframe(df[['Predictions', 'SHAP_sum']])

            # Create a summary plot
            # st.subheader("SHAP Summary Plot")
            # shap.summary_plot(shap_values, feature_names=preprocessor.get_feature_names_out(), show=False)

            # Calculate the average SHAP values for each feature across all rows
            # avg_shap_values = shap_values.values.mean(axis=0)  # Average SHAP value per feature

            # Retrieve the feature names
            

            # Display the average SHAP values and corresponding feature names
            # st.subheader("Average SHAP Values for Each Feature")
            # for feature_name, avg_shap_value in zip(feature_names, avg_shap_values):
            #     st.write(f"Feature: {feature_name} - Average SHAP Value: {avg_shap_value}")

if __name__ == "__main__":
    st.title("SHAP Values Prediction")
    predict_from_csv()
