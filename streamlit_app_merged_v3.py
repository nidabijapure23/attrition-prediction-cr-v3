import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import requests
from category_encoders import TargetEncoder

# Set page config
st.set_page_config(
    page_title="Attrition Prediction App",
    page_icon="📊",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.title("📊 Attrition Prediction")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This app predicts employee attrition using machine learning. Upload your employee data to:
    - Predict attrition risk
    - Analyze risk factors
    - Get tenure predictions
    - Download detailed results
    """)
    
    st.markdown("---")
    st.markdown("### Instructions")
    st.markdown("""
    1. Upload your employee CSV file
    2. Click 'Predict' to run the analysis
    3. View the results and download predictions
    """)
    
    st.markdown("---")
    st.markdown("### Risk Levels")
    st.markdown("""
    - **Severe**: ≥ 90% probability
    - **More Likely**: ≥ 80% probability
    - **Intermediate**: ≥ 64% probability
    - **Mild**: ≥ 50% probability
    - **Minimal**: < 50% probability
    """)

# Load model
try:
    pipeline = joblib.load('xgboost_model_new.pkl')
    # tenure_pipeline = joblib.load('tenure_model_new.pkl')
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Define feature categories (No Tenure)
ordinal_features = ['Education', 'Overall Manager Rating', 'Employee Category', 'Management Level']
nominal_features = ['Gender', 'Work Shift', 'Employee Type', 'Time Type', 'Job Profile']
numerical_features = ['Age_class','Time in Position', 'Years with Current Manager',
                     'Years since Last Promotion', 'Total Base Pay Amount', 'Last Base Pay Increase - Percent',
                     'Scheduled Weekly Hours', 'Companies Worked Count', 'Work Mode', 'Job Level']

# Define ordinal categories
education_levels = ['High School', 'GED', 'Associates', 'Bachelors', 'Masters', 'Doctorate', '']
rating_levels = ['','Not Meeting Expectations', 'Below Expectations', 'Meets Expectations', 'Exceeds Expectation']
employee_category_levels = ['','Agents', 'Office', 'TL', 'MC', 'Management']
management_levels = ['','Staff', 'Middle Management', 'Senior Executives']


# Preprocessing pipelines
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('encoder', OrdinalEncoder(categories=[education_levels, rating_levels, employee_category_levels, management_levels],
                              handle_unknown='use_encoded_value', unknown_value=-1))
])

nominal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_features + ['Cost Center Encoded']),
    ('ord', ordinal_transformer, ordinal_features),
    ('nom', nominal_transformer, nominal_features)
])

# Tenure regressor pipeline
# tenure_numerical_features = [f for f in numerical_features if f != 'Tenure']
# tenure_preprocessor = ColumnTransformer(transformers=[
#     ('num', numerical_transformer, tenure_numerical_features),
#     ('ord', ordinal_transformer, ordinal_features),
#     ('nom', nominal_transformer, nominal_features)
# ])
# tenure_pipeline = Pipeline(steps=[
#     ('preprocessor', tenure_preprocessor),
#     ('regressor', RandomForestRegressor(random_state=42))
# ]) 
# Risk bucketing function
def bucketize_risk(prob):
    if prob >= 0.9:
        return 'Severe'
    elif prob >= 0.8:
        return 'More Likely'
    elif prob >= 0.64:
        return 'Intermediate Risk'
    elif prob >= 0.5:
        return 'Mild Risk'
    else:
        return 'Minimal Risk'

def calculate_tenure(row):
    current_date = pd.to_datetime('2025-04-21')
    if pd.notna(row['Termination Date - All']):
        return (row['Termination Date - All'] - row['Hire Date']).days / 30
    else:
        return (current_date - row['Hire Date']).days / 30

def process_predictions(df):
    # Create target variable if available (1 for left, 0 for stayed)
    if 'Active Status' in df.columns and 'Termination Date - All' in df.columns:
        df['Attrition'] = np.where((df['Active Status'] == 'Yes') & (df['Termination Date - All'].isna()), 0, 1)
    # if 'Hire Date' in df.columns:
    df['Hire Date'] = pd.to_datetime(df['Hire Date'], format='%d-%m-%Y', errors='coerce')
    # if 'Termination Date - All' in df.columns:
    df['Termination Date - All'] = pd.to_datetime(df['Termination Date - All'], format='%d-%m-%Y', errors='coerce')
    # if 'Tenure' not in df.columns and 'Hire Date' in df.columns:
    df['Tenure'] = df.apply(calculate_tenure, axis=1)
    
    # Prepare features for prediction
    feature_cols = numerical_features + ordinal_features + nominal_features + ['Cost Center']
    # print("Required features:", feature_cols)
    # print("Missing features:", [col for col in feature_cols if col not in df.columns])
    
    X = df[feature_cols].copy()
    y = df['Attrition']
    # print("Feature matrix shape:", X.shape)
    
    cost_center_encoder = TargetEncoder(cols=['Cost Center'], smoothing=1)  # Adjust smoothing if needed
    cost_center_encoder.fit(X[['Cost Center']], y)  # Fit to your training data and target
    X_unseen_encoded = cost_center_encoder.transform(X[['Cost Center']])

    X['Cost Center Encoded'] = X_unseen_encoded['Cost Center']
    # Remove original column
    X = X.drop('Cost Center', axis=1)
    # Transform features using the pipeline's preprocessor
    try:
        X_transformed = pipeline.named_steps['preprocessor'].transform(X)
        # print("Transformed features shape:", X_transformed.shape)
        
        # Get predictions
        proba = pipeline.named_steps['classifier'].predict_proba(X_transformed)[:, 1]
        # print("Prediction probabilities shape:", proba.shape)
        # print("Sample probabilities:", proba[:5])
        
        df['Attrition Probability'] = proba
        df['Attrition Prediction'] = np.where(proba >= 0.5, 'Possible Attrition', 'Stayed')
        # df['Attrition Prediction'] = pipeline.predict(X_transformed)
        # df['Attrition Prediction'] = pipeline.named_steps['classifier'].predict(X_transformed)
        df['Risk Level'] = df['Attrition Probability'].apply(bucketize_risk)
        
    except Exception as e:
        print("Error during prediction:", str(e))
        raise e
    
    # Calculate tenure prediction
    # if 'Tenure' not in df.columns:
    #     df['Predicted Tenure'] = tenure_pipeline.predict(X)
    # else:
    # X['Tenure'] = df['Tenure'].loc[X.index]
    # X = X.drop('Tenure', axis=1)
    # df['Predicted Tenure'] = tenure_pipeline.predict(X)
    
    # Calculate SHAP values for feature importance
    try:
        explainer = shap.TreeExplainer(pipeline.named_steps['classifier'])
        shap_values = explainer.shap_values(X_transformed)
        nom_features = pipeline.named_steps['preprocessor'].named_transformers_['nom'].named_steps['encoder'].get_feature_names_out(nominal_features)
        all_features = numerical_features + ordinal_features + list(nom_features)
        triggers = []
        for i in range(X.shape[0]):
            if df.iloc[i]['Attrition Probability'] >= 0.4:
                shap_vals = shap_values[i]
                top_indices = np.argsort(np.abs(shap_vals))[-5:][::-1]
                triggers.append(', '.join([all_features[j] for j in top_indices]))
            else:
                triggers.append('')
        df['Triggers'] = triggers
    except Exception as e:
        print('SHAP error:', e)
        df['Triggers'] = ''
    
    # Create Actual Status column
    def get_actual_status(row):
        if str(row.get('Active Status')).strip().lower() == 'yes':
            return 'Yes'
        if 'Termination Date - All' in row and pd.notna(row.get('Termination Date - All')):
            return 'No'
        return 'No'
    df['Actual Status'] = df.apply(get_actual_status, axis=1)
    
    # print("Final DataFrame shape:", df.shape)
    # print("Final columns:", df.columns.tolist())
    
    return df

def save_false_positives_to_excel(false_positives_df, filename='false_positives_tracking.xlsx'):
    """
    Save false positive predictions to a centralized Excel file.
    If file exists, append new data; if not, create new file.
    """
    # Add timestamp column
    false_positives_df['Prediction_Date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Initialize comment columns if they don't exist
    if 'HR_Comments' not in false_positives_df.columns:
        false_positives_df['HR_Comments'] = ''
    if 'OPS_comments' not in false_positives_df.columns:
        false_positives_df['OPS_comments'] = ''
    
    # Select and order columns for saving
    save_cols = ['Employee ID', 'Attrition Prediction', 'Attrition Probability', 'Risk Level', 'Triggers', 
                'Prediction_Date', 'HR_Comments', 'OPS_comments', 'Cost Center']
    
    try:
        # Try to read existing file
        existing_df = pd.read_excel(filename)
        # Combine existing and new data
        combined_df = pd.concat([existing_df, false_positives_df[save_cols]], ignore_index=True)
        # Remove any duplicates based on Employee ID and Prediction_Date
        combined_df = combined_df.drop_duplicates(subset=['Employee ID', 'Prediction_Date'])
        # Save back to Excel
        combined_df.to_excel(filename, index=False)
        return True
    except FileNotFoundError:
        # If file doesn't exist, create new file
        false_positives_df[save_cols].to_excel(filename, index=False)
        return True
    except Exception as e:
        print(f"Error saving false positives: {str(e)}")
        return False

# Main content
st.title("📊 Attrition Prediction App")

# Add template download
st.subheader("📋 Sample Input File")
try:
    # Read the sample input file
    sample_df = pd.read_csv('sample_input_data.csv')
    
    # Display the sample data
    st.write("Below is a sample input file with required headers and example data:")
    # st.dataframe(sample_df, use_container_width=True)
    
    # Provide download option
    with open('sample_input_data.csv', 'rb') as f:
        st.download_button(
            label="Download Sample Input File",
            data=f,
            file_name='sample_input_data.csv',
            mime='text/csv',
            help="Click to download the sample input file"
        )
except Exception as e:
    st.error(f"Error loading sample input file: {str(e)}")
    st.info("Please ensure 'sample_input_data.csv' exists in the application directory")

# File upload
uploaded_file = st.file_uploader("Upload Employee CSV", type="csv")

if uploaded_file:
    st.session_state.uploaded_file = uploaded_file
    st.write("File uploaded. Click Predict to run analysis.")
    if st.button("Predict"):
        with st.spinner('Running prediction...'):
            # Read and process the file
            df = pd.read_csv(uploaded_file, encoding='latin-1')
            
            # Process predictions
            df = process_predictions(df)
            
            # Display summary
            st.subheader("📈 Prediction Summary")
            
            # Calculate summary statistics
            total_active = (df['Attrition Prediction'] == 'Stayed').sum()
            total_inactive = (df['Attrition Prediction'] == 'Possible Attrition').sum()
            risk_level_counts = df[df['Attrition Prediction'] == 'Possible Attrition'].groupby('Risk Level').size().to_dict()
            
            # Create summary table
            metrics = ["Total Active Predictions", "Total Inactive Predictions"]
            counts = [total_active, total_inactive]
            
            # Add risk level counts to summary
            for risk_level, count in risk_level_counts.items():
                metrics.append(f"{risk_level} ⚠️")
                counts.append(count)
            
            summary_data = {
                "Metric": metrics,
                "Count": counts
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df = summary_df.sort_values(by='Count', ascending=False)
            summary_df.index = summary_df.index + 1
            summary_df.index.name = "SR.No."

            # Graphical risk level summary
            def highlight_risk(row):
                color = ''
                if 'Severe' in row['Metric']:
                    color = 'color: #ff4d4d; font-weight: bold;'  # Red
                elif 'More Likely' in row['Metric']:
                    color = 'color: #ff944d; font-weight: bold;'  # Orange
                elif 'Intermediate' in row['Metric']:
                    color = 'color:rgb(255, 190, 77); font-weight: bold;'  # Yellow
                elif 'Mild' in row['Metric']:
                    color = 'color:rgb(205, 205, 33);'  # Light yellow
                elif 'Minimal' in row['Metric']:
                    color = 'color:rgb(246, 246, 78);' 
                elif 'Total Active Predictions' in row['Metric']:
                    color = 'color:rgb(5, 115, 27);'  
                elif 'Total Inactive Predictions' in row['Metric']:
                    color = 'color:rgb(209, 28, 28);' 
                else:
                    color = ''
                return [color] * len(row)

            
            st.dataframe(summary_df.style.apply(highlight_risk, axis=1), use_container_width=True)
            
            # Display false positives (active employees predicted as attrite)
            st.subheader('⚠️ Active Employees Predicted as "At Risk"')
            false_positives = df[(df['Attrition Prediction'] == 'Possible Attrition') & (df['Actual Status'] == 'Yes')]
            if not false_positives.empty:
                # Sort by attrition probability descending
                false_positives = false_positives.sort_values(by='Attrition Probability', ascending=False)
                false_positives = false_positives.reset_index(drop=True)
                false_positives.index = false_positives.index + 1
                false_positives.index.name = "SR.No."
                # Calculate Actual Tenure
                false_positives['Actual Tenure'] = false_positives.apply(calculate_tenure, axis=1)
                # Calculate Variation
                false_positives['Variation'] = false_positives['Predicted Tenure'] - false_positives['Actual Tenure']
                display_cols = ['Employee ID', 'Attrition Prediction', 'Risk Level', 'Triggers']
                st.dataframe(false_positives[display_cols], use_container_width=True)
                
                # Save false positives to centralized tracking file
                if save_false_positives_to_excel(false_positives):
                    st.success("Predictions have been saved to the tracking file.")
                else:
                    st.warning("Failed to save false positive predictions to tracking file.")
            else:
                st.success("No active employees predicted as at risk!")

            # Save and provide download for all predictions
            st.subheader("📥 Download Predictions")
            
            # Add SR.No. to all predictions
            all_predictions = df.copy()
            all_predictions = all_predictions.reset_index(drop=True)
            all_predictions.index = all_predictions.index + 1
            all_predictions = all_predictions.reset_index().rename(columns={'index': 'SR.No.'})
            
            # Calculate Actual Tenure and Variation
            all_predictions['Actual Tenure'] = all_predictions.apply(calculate_tenure, axis=1)
            all_predictions['Variation'] = all_predictions['Predicted Tenure'] - all_predictions['Actual Tenure']
            
            # Select columns for output
            output_cols = ['SR.No.', 'Employee ID', 'Attrition Prediction', 'Risk Level', 'Attrition Probability', 
                         'Triggers', 'Actual Status']
            
            all_predictions[output_cols].to_csv('all_predictions.csv', index=False)
            with open('all_predictions.csv', 'rb') as f:
                st.download_button(
                    label="Download All Predictions",
                    data=f,
                    file_name='all_predictions.csv',
                    mime='text/csv',
                    help="Click to download the complete predictions file"
                )
            
            
