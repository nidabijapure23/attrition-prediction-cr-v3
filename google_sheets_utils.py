import pandas as pd
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import streamlit as st

def get_google_sheets_credentials():
    """Get Google Sheets credentials from Streamlit secrets"""
    try:
        # Debug: Print available secrets keys
        # st.write("Available secrets keys:", list(st.secrets.keys()))
        
        # Get the service account info from secrets
        service_account_info = {
            "type": st.secrets["gcp_service_account"]["type"],
            "project_id": st.secrets["gcp_service_account"]["project_id"],
            "private_key_id": st.secrets["gcp_service_account"]["private_key_id"],
            "private_key": st.secrets["gcp_service_account"]["private_key"].replace("\\n", "\n"),
            "client_email": st.secrets["gcp_service_account"]["client_email"],
            "client_id": st.secrets["gcp_service_account"]["client_id"],
            "auth_uri": st.secrets["gcp_service_account"]["auth_uri"],
            "token_uri": st.secrets["gcp_service_account"]["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["gcp_service_account"]["auth_provider_x509_cert_url"],
            "client_x509_cert_url": st.secrets["gcp_service_account"]["client_x509_cert_url"]
        }
        
        # Debug: Print service account email
        # st.write("Service Account Email:", service_account_info["client_email"])
        
        return Credentials.from_service_account_info(
            service_account_info,
            scopes=["https://www.googleapis.com/auth/spreadsheets"]
        )
    except Exception as e:
        st.error(f"Error loading Google Sheets credentials: {str(e)}")
        st.error("Full error details:", str(e.__class__.__name__))
        return None

def get_sheet_data(spreadsheet_id, range_name):
    """Read data from Google Sheet"""
    try:
        creds = get_google_sheets_credentials()
        if not creds:
            return None
            
        service = build('sheets', 'v4', credentials=creds)
        sheet = service.spreadsheets()
        
        result = sheet.values().get(
            spreadsheetId=spreadsheet_id,
            range=range_name
        ).execute()
        
        values = result.get('values', [])
        if not values:
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame(values[1:], columns=values[0])
        return df
        
    except Exception as e:
        st.error(f"Error reading from Google Sheet: {str(e)}")
        return None

def update_sheet_data(spreadsheet_id, range_name, df):
    """Update data in Google Sheet"""
    try:
        creds = get_google_sheets_credentials()
        if not creds:
            return False
            
        service = build('sheets', 'v4', credentials=creds)
        sheet = service.spreadsheets()
        
        # First, clear the existing data in the range
        sheet.values().clear(
            spreadsheetId=spreadsheet_id,
            range=range_name
        ).execute()
        
        # Convert DataFrame to list of lists
        values = [df.columns.tolist()] + df.values.tolist()
        
        body = {
            'values': values
        }
        
        # Update with new data
        result = sheet.values().update(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            valueInputOption='USER_ENTERED',  # Changed from 'RAW' to 'USER_ENTERED'
            body=body
        ).execute()
        
        return True
        
    except Exception as e:
        st.error(f"Error updating Google Sheet: {str(e)}")
        return False

def append_sheet_data(spreadsheet_id, range_name, df):
    """Append data to Google Sheet"""
    try:
        creds = get_google_sheets_credentials()
        if not creds:
            return False
            
        service = build('sheets', 'v4', credentials=creds)
        sheet = service.spreadsheets()
        
        # Convert DataFrame to list of lists
        values = df.values.tolist()
        
        body = {
            'values': values
        }
        
        result = sheet.values().append(
            spreadsheetId=spreadsheet_id,
            range=range_name,
            valueInputOption='RAW',
            body=body
        ).execute()
        
        return True
        
    except Exception as e:
        st.error(f"Error appending to Google Sheet: {str(e)}")
        return False 