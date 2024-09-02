import streamlit as st
import sys
import os
import argparse
from pathlib import Path
from glob import glob
import pydicom
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import yaml
import streamlit_authenticator as stauth
from hash_password import hide_users

# Load the users.yml and selects.yml files from environment variables
users_yml_path = os.getenv('USERS_YML', 'Users.yml')
selects_yml_path = os.getenv('SELECTS_YML', 'select.yml')

# users_yml_path = 'Users.yml'
# selects_yml_path =  'select.yml'
# file_path = 'study_samples'

def apply_window_level(dicom_array, window_width, window_level):
    """Apply window level to the DICOM image array."""
    lower_bound = window_level - (window_width / 2)
    upper_bound = window_level + (window_width / 2)
    
    dicom_array = np.clip(dicom_array, lower_bound, upper_bound)
    dicom_array = (dicom_array - lower_bound) / (upper_bound - lower_bound) * 255.0
    return dicom_array.astype(np.uint8)

def load_dicom_image(dicom_path):
    """Load a DICOM file and return the pixel array, window width, and window level."""
    dicom = pydicom.dcmread(dicom_path)
    dicom_array = dicom.pixel_array
    window_width = dicom.WindowWidth if 'WindowWidth' in dicom else dicom_array.max() - dicom_array.min()
    window_level = dicom.WindowLevel if 'WindowLevel' in dicom else (dicom_array.max() + dicom_array.min()) / 2
    return dicom_array, window_width, window_level


def convert_df(dataframe: pd.DataFrame):
    return dataframe.to_csv(index=False).encode('utf-8')


def save_annotations(csv_path, annotations):
    """Save annotations to a CSV file."""

    if not os.path.isfile(csv_path):
        df = pd.DataFrame(annotations)
        df.to_csv(csv_path, index=False)
    else:
        orig_df = pd.read_csv(csv_path)*1
        df = pd.DataFrame(annotations)

        orig_df.loc[orig_df.ID.isin(df.ID.astype(int)),df.keys()] = df[df.ID.isin(orig_df.ID.astype(int))]*1

        orig_df.to_csv(csv_path, index=False)

        

def write_csv(csv_path, selects, ids):
    if not os.path.isfile(csv_path):
        os.makedirs(csv_path.parent, exist_ok=True)
        with open(str(csv_path), 'w+') as results_file:
            string = 'ID, annotated,text_comment,'.replace(
                                   " ",""
                               ) + ",".join([
                                   ",".join([f"{elem},certainty_{elem}" for elem in selects[key]])
                                   for key in selects.keys()
                                   ]) + "\n"
            results_file.write(string)

            elementstring = ",".join([
                                   ",".join(["," for elem in selects[key]])
                                   for key in selects.keys()
                                   ])
            
            for id in ids:
                results_file.writelines(f'{id},{0},{elementstring}\n')

def main(file_path: str, file_type: str, study_type: str):

    print(os.path.isfile(users_yml_path))
    print(os.path.isfile(selects_yml_path))

    hide_users()

    # Load the users.yml file
    with open(users_yml_path, 'r') as file:
        config = yaml.safe_load(file)

    # Load the select.yml file
    with open(selects_yml_path, 'r') as file:
        selections = yaml.safe_load(file)

    # Get user credentials from the config file
    credentials = config['credentials']
    
    # Set up the authenticator
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )

    # Create login widget
    name, authentication_status, username = authenticator.login('main', fields = {'Form name': 'Rayvolve Study Login'})


    # If authentication is successful, run the main app
    if authentication_status:

        start_id = credentials["usernames"][name]['startid']
        end_id = credentials["usernames"][name]['endid']

        # Set up the logout button
        if st.session_state.get("authentication_status"):
            authenticator.logout('Logout', 'sidebar')
        
        current_file_path = "results"
        
        # Get a list of all files in the main file

        ray_ids = glob(str(Path(file_path) / '*'))
        
        ray_ids = sorted(list({
            int(Path(file).stem.split("_")[0][3:]) for file in ray_ids
        }))

        ray_ids = [index for index in ray_ids if (start_id<=index and end_id>=index) ] 

        files = {
            index: glob(str(Path(file_path) / f"RAY{index:03}" / f'*_{study_type}.{file_type}'))
            for index in ray_ids
        }

        csv_path = Path(__file__).parent / 'results' / f'{username}-results.csv'


        write_csv(csv_path, selections, ray_ids)

        df = pd.read_csv(str(csv_path))

        done = len(files)==0

        
        try:
            try:
                probe = df[~(df['annotated']==1)].ID.tolist()[0]
            except IndexError:
                done = True
                st.write('No more files! You are done!')
                st.balloons()
        except AttributeError:
            raise AttributeError(f'Data directory does not contain any images')

        if not done:

            # Initialize session state to keep track of the current file index
            if 'file_index' not in st.session_state:
                st.session_state.file_index = 0

            # Define functions to handle next/prev button clicks
            def next_file():
                if st.session_state.file_index < len(files) - 1:
                    st.session_state.file_index += 1

            def prev_file():
                if st.session_state.file_index > 0:
                    st.session_state.file_index -= 1

            # Display navigation buttons
            col1, col2 = st.columns([1, 1])
            with col1:
                st.button("Previous", on_click=prev_file)
            with col2:
                st.button("Next", on_click=next_file)

            progress_bar = st.progress(int(((df.annotated.sum() / len(files))) * 100),
                                    text=f'{int(((df.annotated.sum() / len(files))) * 100)} %')

            st.header(f"file: {st.session_state.file_index + 1}/{len(files)}")

            # Get the current file name
            current_files = files[ray_ids[st.session_state.file_index]]

            # Display the current file name

            columns = st.columns(len(current_files))
            # Loop through all DICOM files in the file
            for i, dicom_path in enumerate(current_files):
                with columns[i]:
                    dicom_file = Path(dicom_path).stem

                    # Load the DICOM image and apply window/level adjustments
                    dicom_array, default_window_width, default_window_level = load_dicom_image(dicom_path)

                    # Create sliders for window width and window level
                    window_width = st.slider(f"Window Width_{i}", min_value=1, max_value=int(dicom_array.max()), value=int(default_window_width))
                    window_level = st.slider(f"Window Level_{i}", min_value=int(dicom_array.min()), max_value=int(dicom_array.max()), value=int(default_window_level))

                    # Apply window/level and display the image
                    dicom_array = apply_window_level(dicom_array, window_width, window_level)
                    image = Image.fromarray(dicom_array)
                    st.image(image, use_column_width=True)

            category = Path(current_files[0]).stem.split("_")[1]

            with st.expander("### Aufgabenerklaerung:", expanded=True):
                st.header(" Bitte klicken Sie alles an was zutrifft. \n ### Daraufhin erscheint ein Slider. Geben Sie bitte durch diesen Slider an, wie sicher Sie sich dabei sind. \n ### Nachdem Sie Ihre Bewertung angegeben haben, klicken Sie bitte auf 'Submit'.")

            conditions = {}
            for subselects in selections[category]:
                conditions[subselects] = st.checkbox(f"{subselects}", key=f"checkbox_{subselects}")
                if conditions[subselects]:
                    # Slider to indicate certainty level
                    conditions["certainty_"+subselects] = st.slider(f"Certainty Level", min_value=1, max_value=5, value=3, key=f"slider_{subselects}")


            text_comment = st.text_input("Kommentarfeld:", )

            # Store the annotation
            annotations = [{
                "ID": ray_ids[st.session_state.file_index],
                "text_comment": text_comment,
                "annotated": True
            } | conditions]

            enter_results = st.button("Submit")

            if enter_results:
                save_annotations(csv_path, annotations)
                next_file()
                st.rerun()

        with open(csv_path, "rb") as f:
            st.download_button(
                label="Download Annotations CSV",
                data=f,
                file_name=f"{username}-results.csv",
                mime="text/csv"
            )

    elif authentication_status == False:
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')


if __name__ == "__main__":
    file_path = os.getenv('FILE_PATH', '')

    mode = os.getenv("STUDY_MODE", "original")

    if not file_path:
        st.error("FILE_PATH environment variable must be set.")
    else:
        main(file_path, "dcm", mode)