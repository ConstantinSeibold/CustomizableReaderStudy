import os
from pathlib import Path
from glob import glob
import json
import streamlit as st
import yaml
import pandas as pd
import numpy as np
from PIL import Image
import streamlit_authenticator as stauth
from typing import List, Dict, Tuple
import streamlit.components.v1 as components
from hash_password import hide_users  # Assuming you have this

# Set page configuration
st.set_page_config(layout="wide")

# Environment variable fallbacks
users_yml_path: str = os.getenv('USERS_YML', 'Users.yml')
assignment_csv_path: str = os.getenv('ASSIGNMENT_CSV', 'users.csv')
selects_yml_path: str = os.getenv('SELECTS_YML', 'select.yml')
descriptor_yml_path: str = os.getenv('DESCRIPTOR_YML', 'descriptor.yml')
tmp_storage_path: str = os.getenv('TMP_STORAGE', 'files.json')


# Function for authenticating the user
def authenticate_user() -> Tuple[Dict, stauth.Authenticate]:
    """
    Handles user authentication and loading of necessary configuration files.

    Returns:
        descriptor_yml (Dict): Loaded YAML configuration for descriptors.
        authenticator (stauth.Authenticate): Authenticator instance for handling user authentication.
    """
    # Hide sensitive user details
    hide_users()

    # Load the users.yml configuration
    with open(users_yml_path, 'r') as file:
        config: Dict = yaml.safe_load(file)

    # Load the descriptor.yml configuration
    with open(descriptor_yml_path, 'r') as file:
        descriptor_yml: Dict = yaml.safe_load(file)

    # Extract credentials and cookie settings
    credentials: Dict = config['credentials']

    # Initialize the authenticator with credentials and cookie configuration
    authenticator: stauth.Authenticate = stauth.Authenticate(
        credentials,
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
    )

    # Display the login form with custom field labels from the descriptor file
    try:
        authenticator.login('main', fields={
            'Form name': descriptor_yml["login"]["study_name"],
            'Username': descriptor_yml["login"]["username"],
            'Password': descriptor_yml["login"]["password"],
            'Login': descriptor_yml["login"]["login"]
        })
    except Exception as e:
        st.error(f"Login error: {e}")

    return descriptor_yml, authenticator


# Main application logic
def main() -> None:
    """
    Main function to handle the user interface and navigation after authentication.
    """
    descriptor_yml, authenticator = authenticate_user()

    # Retrieve the authentication status from session state
    authentication_status = st.session_state.get('authentication_status')

    if authentication_status:
        # Logout button in the sidebar
        authenticator.logout(descriptor_yml["login"]["logout"], 'sidebar')
        st.session_state["authenticator"] = authenticator
        # Redirect to study page
        st.switch_page("pages/study.py")

    elif authentication_status is False:
        # Display error message for failed authentication
        st.error(descriptor_yml["login"]["error"])

    elif authentication_status is None:
        # Display warning message if authentication is incomplete
        st.warning(descriptor_yml["login"]["warning"])


# Run the main function
if __name__ == "__main__":
    main()
