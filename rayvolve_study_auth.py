import os
from pathlib import Path
from glob import glob

import streamlit as st
import pydicom
import numpy as np
import pandas as pd
from PIL import Image
import yaml
import streamlit_authenticator as stauth
from hash_password import hide_users



# Load the paths to the users.yml and selects.yml configuration files from environment variables.
# Defaults are provided in case the environment variables are not set.
users_yml_path: str = os.getenv('USERS_YML', 'Users.yml')
selects_yml_path: str = os.getenv('SELECTS_YML', 'select.yml')
descriptor_yml_path: str = os.getenv('DESCRIPTOR_YML', 'descriptor.yml')

file_path: str = os.getenv('FILE_PATH', '')
mode: str = os.getenv("STUDY_MODE", "original")
file_type: str = os.getenv("FILE_TYPE", "dcm")

# Check existence of the configuration files.
if not (os.path.isfile(users_yml_path) and \
        os.path.isfile(selects_yml_path) and\
        os.path.isfile(descriptor_yml_path)):
    st.error("Configuration files are missing.")

# Hide user credentials from the environment.
hide_users()

# Load the users.yml file
with open(users_yml_path, 'r') as file:
    config = yaml.safe_load(file)

# Load the select.yml file
with open(selects_yml_path, 'r') as file:
    selections = yaml.safe_load(file)

# Load the select.yml file
with open(descriptor_yml_path, 'r') as file:
    descriptor_yml = yaml.safe_load(file)

ray_ids = glob(str(Path(file_path) / '*'))
        
ray_ids = sorted(list({
    int(Path(file).stem.split("_")[0][3:]) for file in ray_ids
}))

prefix = descriptor_yml["study_prefix"]

all_files = {
            index: sum(
                    [
                        glob(str(Path(file_path) / f"{prefix}{index:03}" / f'*_{st}.{file_type}')) 
                        for st in mode.split("|")
                    ], 
                    []
                    )
            for index in ray_ids 
        }

# users_yml_path = 'Users.yml'
# selects_yml_path =  'select.yml'
# file_path = 'study_samples'

def apply_window_level(dicom_array: np.ndarray, window_width: float, window_level: float) -> np.ndarray:
    """Applies window leveling to a DICOM image array.

    Args:
        dicom_array (np.ndarray): The DICOM image pixel array.
        window_width (float): The width of the window.
        window_level (float): The level of the window.

    Returns:
        np.ndarray: The window-leveled image array as an 8-bit unsigned integer.
    """
    lower_bound: float = window_level - (window_width / 2)
    upper_bound: float = window_level + (window_width / 2)
    
    # Clip and scale the pixel array to the [0, 255] range.
    dicom_array = np.clip(dicom_array, lower_bound, upper_bound)
    dicom_array = ((dicom_array - lower_bound) / (upper_bound - lower_bound)) * 255.0
    return dicom_array.astype(np.uint8)


def load_dicom_image(dicom_path: str) -> tuple[np.ndarray, float, float]:
    """Loads a DICOM file and extracts the pixel array along with window width and level.

    Args:
        dicom_path (str): Path to the DICOM file.

    Returns:
        tuple: A tuple containing the DICOM image array, window width, and window level.
    """
    dicom = pydicom.dcmread(dicom_path)
    dicom_array = dicom.pixel_array
    window_width: float = dicom.WindowWidth if 'WindowWidth' in dicom else dicom_array.max() - dicom_array.min()
    window_level: float = dicom.WindowLevel if 'WindowLevel' in dicom else (dicom_array.max() + dicom_array.min()) / 2
    return dicom_array, window_width, window_level


def convert_df(dataframe: pd.DataFrame) -> bytes:
    """Converts a DataFrame to a CSV byte string.

    Args:
        dataframe (pd.DataFrame): The DataFrame to convert.

    Returns:
        bytes: The CSV data encoded as UTF-8 bytes.
    """
    return dataframe.to_csv(index=False).encode('utf-8')

def save_annotations(csv_path: str, annotations: list[dict]) -> None:
    """Saves annotations to a CSV file, updating existing entries if present.

    Args:
        csv_path (str): Path to the CSV file.
        annotations (list[dict]): List of annotation dictionaries to save.
    """
    df = pd.DataFrame(annotations)

    if not os.path.isfile(csv_path):
        # Create new CSV if it doesn't exist.
        df.to_csv(csv_path, index=False)
    else:
        # Update existing annotations.
        orig_df = pd.read_csv(csv_path)

        keys = list(set(df.keys())-{"ID"})
        indices = orig_df.ID.astype(int)==df.iloc[0].ID.astype(int)
        
        orig_df.loc[indices,keys] = df.loc[[0],keys].values
        # orig_df.loc[orig_df.ID.astype(int)==df.iloc[0].ID.astype(int),df.keys()] = df.iloc[0]

        orig_df.to_csv(csv_path, index=False)

        

def write_csv(csv_path: Path, selects: dict, ids: list[int]) -> None:
    """Writes the initial CSV structure for annotations.

    Args:
        csv_path (Path): Path to save the CSV file.
        selects (dict): Dictionary of selections for annotations.
        ids (list[int]): List of IDs to include in the CSV.
    """
    if not os.path.isfile(csv_path):
        os.makedirs(csv_path.parent, exist_ok=True)
        with open(str(csv_path), 'w+') as results_file:
            header = 'ID, annotated,text_comment,'.replace(
                                   " ",""
                               ) + ",".join([
                                   ",".join([f"{elem},certainty_{elem}" for elem in selects[key]])
                                   for key in selects.keys()
                                   ]) + "\n"
            results_file.write(header)

            row_template = "Nothing,"+",".join([
                                   ",".join(["False,-1" for elem in selects[key]])
                                   for key in selects.keys()
                                   ])
            for id in ids:
                results_file.writelines(f'{id},{False},{row_template}\n')
                

def main() -> None:
    """Main application logic.

    Args:
        file_path (str): Path to the study files.
        file_type (str): Type of the file (e.g., 'dcm').
        study_type (str): Type of study (e.g., 'original').
    """
    
    # Get user credentials from the config file and set up the authenticator
    credentials = config['credentials']
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )

    # Create login widget
    name, authentication_status, username = authenticator.login('main', fields = {
        'Form name': descriptor_yml["login"]["study_name"],
        'Username': descriptor_yml["login"]["username"], 
        'Password': descriptor_yml["login"]["password"], 
        'Login': descriptor_yml["login"]["login"]
        })


    # If authentication is successful, run the main app
    if authentication_status:

        start_id = credentials["usernames"][name]['startid']
        end_id = credentials["usernames"][name]['endid']

        # Set up the logout button
        if st.session_state.get("authentication_status"):
            authenticator.logout(descriptor_yml["login"]["logout"], 'sidebar')
        
        current_file_path = "results"   
        
        ray_ids = [index for index in ray_ids if (start_id<=index and end_id>=index) ] 

        # Get a list of all files in the main file
        
        files = {
            index: all_files[index]
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
                st.button(descriptor_yml["selection"]["previous"], on_click=prev_file)
            with col2:
                st.button(descriptor_yml["selection"]["next"], on_click=next_file)

            progress_bar = st.progress(int(((df.annotated.sum() / len(files))) * 100),
                                    text=f'{int(((df.annotated.sum() / len(files))) * 100)} %')

            
            with st.expander(descriptor_yml["task"]["task_name"], expanded=True):
                info_md = descriptor_yml["task"]["task_caption"]
                st.markdown(info_md, unsafe_allow_html=True)
                
            st.header(descriptor_yml["case"] + f" {st.session_state.file_index + 1}/{len(files)}")

            

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
                    window_width = st.slider(f"Window Width_{i}", 
                                            min_value=1, 
                                            max_value=int(dicom_array.max()), 
                                            value=int(default_window_width)
                                            )
                    window_level = st.slider(f"Window Level_{i}", 
                                            min_value=int(dicom_array.min()),
                                            max_value=int(dicom_array.max()),
                                            value=int(default_window_level)
                                            )

                    # Apply window/level and display the image
                    dicom_array = apply_window_level(dicom_array, window_width, window_level)
                    image = Image.fromarray(dicom_array)
                    st.image(image, use_column_width=True)

            category = Path(current_files[0]).stem.split("_")[1]            

            conditions = {}
            for subselects in selections[category]:
                conditions[subselects] = st.checkbox(
                                                     f"{subselects}", 
                                                     key=f"checkbox_{subselects}", 
                                                     value=False
                                                     )
                if conditions[subselects]:
                    # Slider to indicate certainty level
                    conditions["certainty_"+subselects] = st.slider(descriptor_yml["certainty"]["certainty_caption"], 
                                                                    min_value=descriptor_yml["certainty"]["min_certainty"], 
                                                                    max_value=descriptor_yml["certainty"]["max_certainty"], 
                                                                    value=(descriptor_yml["certainty"]["min_certainty"]+descriptor_yml["certainty"]["max_certainty"])//2, 
                                                                    key=f"slider_{subselects}")


            text_comment = st.text_input(descriptor_yml["comments"], )

            # Store the annotation
            annotations = [{
                "ID": ray_ids[st.session_state.file_index],
                "text_comment": text_comment,
                "annotated": True
            } | conditions]

            enter_results = st.button(descriptor_yml["submit"])

            st.write(st.session_state)

            if enter_results:
                save_annotations(csv_path, annotations)
                next_file()
                # https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state
                st.rerun()

        with open(csv_path, "rb") as f:
            st.download_button(
                label=descriptor_yml["download"],
                data=f,
                file_name=f"{username}-results.csv",
                mime="text/csv"
            )

    elif authentication_status == False:
        st.error(descriptor_yml["login"]["error"])
    elif authentication_status == None:
        st.warning(descriptor_yml["login"]["warning"])


if __name__ == "__main__":

    main()