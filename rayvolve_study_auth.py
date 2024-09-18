import os
from pathlib import Path
from glob import glob
import json
from typing import List, Dict, Tuple

import streamlit as st
import streamlit.components.v1 as components
import pydicom
import numpy as np
import pandas as pd
from PIL import Image
import yaml
import streamlit_authenticator as stauth
from hash_password import hide_users

st.set_page_config(layout="wide")

# Load the paths to the users.yml and selects.yml configuration files from environment variables.
# Defaults are provided in case the environment variables are not set.
users_yml_path: str = os.getenv('USERS_YML', 'Users.yml')
selects_yml_path: str = os.getenv('SELECTS_YML', 'select.yml')
descriptor_yml_path: str = os.getenv('DESCRIPTOR_YML', 'descriptor.yml')
tmp_storage_path: str = os.getenv('TMP_STORAGE', 'files.json')


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


def load_image(image_path: str) -> tuple[np.ndarray, float, float]:
    """
    Loads a image (e.g., PNG, JPEG) and extracts the pixel array along with a calculated window width and level.

    Args:
        image_path (str): Path to the image file.

    Returns:
        tuple: A tuple containing the image array, window width, and window level.
    """
    # Load image using Pillow and convert it to grayscale
    image = Image.open(image_path).convert('L')
    
    # Convert image to numpy array
    image_array = np.array(image)
    
    # Calculate window width and level from pixel values
    window_width: float = image_array.max() - image_array.min()
    window_level: float = (image_array.max() + image_array.min()) / 2
    
    return image_array, window_width, window_level

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


def get_all_files(file_path: str, prefix: str, file_type: str, mode: str, tmp_storage: str) -> Tuple[List[int], Dict[int, List[str]]]:
    """
    Retrieves all files based on specified file path, prefix, and file type.
    
    Args:
        file_path (str): The base directory where files are located.
        prefix (str): The prefix for the files to filter.
        file_type (str): The type of files to search for (e.g., txt, csv).
        mode (str): The file mode or file identifier (can be multiple, separated by '|').
        tmp_storage (str): The path to a temporary storage file.

    Returns:
        Tuple[List[int], Dict[int, List[str]]]: A tuple containing:
            - List of ray_ids (int): Unique IDs extracted from filenames.
            - Dictionary of all files where the key is the ray ID and the value is a list of file paths.
    """
    
    # If the temporary storage file does not exist, fetch files from the directory
    if not os.path.isfile(tmp_storage):
        # Get a list of all files matching the pattern in the given directory
        ray_files = glob(str(Path(file_path) / '*'))

        # Extract unique IDs (numbers after 'ray' prefix) from the filenames
        ray_ids = sorted(list({
            int(Path(file).stem.split("_")[0][3:])  # Extract the numeric part after 'ray'
            for file in ray_files
        }))
        
        # Create a dictionary where each key is a ray_id and the value is a list of matching files
        all_files = {
            index: sum(
                [
                    glob(str(Path(file_path) / f"{prefix}{index:03}" / f'*_{st}.{file_type}'))
                    for st in mode.split("|")
                ], []
            )
            for index in ray_ids
        }

        with open(tmp_storage, 'w') as f:
            json.dump(all_files, f)

    else:
        # Load file data from the tmp_storage JSON file
        with open(tmp_storage, 'r') as f:
            all_files = json.load(f)

        all_files = {int(k): v for k, v in all_files.items()}
        # Convert the dictionary keys (ray_ids) to integers
        ray_ids = all_files.keys()

    # Return the list of ray_ids and the dictionary of all files
    return ray_ids, all_files

def main(file_path: str, file_type: str, study_type: str) -> None:
    """Main application logic.

    Args:
        file_path (str): Path to the study files.
        file_type (str): Type of the file (e.g., 'dcm').
        study_type (str): Type of study (e.g., 'original').
    """
    # Check existence of the configuration files.
    if not (os.path.isfile(users_yml_path) and os.path.isfile(selects_yml_path)):
        st.error("Configuration files are missing.")
        return

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
                
        # Get a list of all files in the main file        
        
        prefix = descriptor_yml["study_prefix"]

        ray_ids, all_files = get_all_files(file_path, prefix, file_type, mode, tmp_storage_path)

        # Filter IDs based on start-id and end-id defined in the Users.yml
        ray_ids = [index for index in ray_ids if (start_id<=index and end_id>=index) ] 

        session_states = {ray_id: i for i,ray_id in enumerate(ray_ids)}

        files = {
            index: all_files[index]
            for index in ray_ids 
        }
        
        csv_path = Path(__file__).parent / 'results' / f'{username}-results.csv'

        write_csv(csv_path, selections, ray_ids)

        df = pd.read_csv(str(csv_path))

        df.annotated = df.annotated.astype(bool)
        done = len(files)==0

        try:
            try:
                probe = df[~(df['annotated']==True)].ID.tolist()[0]
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
                
                components.html(
                    f"""
                        <script>
                            window.parent.document.querySelector('section.main').scrollTo(0, 0);
                        </script>
                    """,
                    height=0,
                )

            def prev_file():
                if st.session_state.file_index > 0:
                    st.session_state.file_index -= 1


                components.html(
                    f"""
                        <script>
                            window.parent.document.querySelector('section.main').scrollTo(0, 0);
                        </script>
                    """,
                    height=0,
                )

            # Display navigation buttons
            col1,  col2, col3 = st.columns([1,3, 1])
            with col1:
                st.button(descriptor_yml["selection"]["previous"], on_click=prev_file)
            with col2:
                with st.expander(descriptor_yml["selection"]["missing"]):
                    max_show_files = descriptor_yml["selection"]["max_missing"]
                    missing_ids = df[~df.annotated].ID.tolist()[:max_show_files]                    
                    missing_cols = st.columns(max_show_files)
                    skipping = {}
                    for i,missing_id in enumerate(missing_ids):
                        with missing_cols[i]:
                            skipping[missing_id] = st.button(f"{session_states[missing_id] + 1}")
                            if skipping[missing_id]:
                                st.session_state.file_index = session_states[missing_id]
            with col3:
                st.button(descriptor_yml["selection"]["next"], on_click=next_file)

            progress_bar = st.progress(int(((df.annotated.sum() / len(files))) * 100),
                                    text=f'{int(((df.annotated.sum() / len(files))) * 100)} %')

            
            with st.expander(descriptor_yml["task"]["task_name"], expanded=True):
                info_md = descriptor_yml["task"]["task_caption"]
                st.markdown(info_md, unsafe_allow_html=True)

            is_annotated = df[df.ID == ray_ids[st.session_state.file_index]].annotated.iloc[0]
                
            st.header(descriptor_yml["case"] + f" {st.session_state.file_index + 1}/{len(files)} - " + ("☑" if is_annotated else "☐") )

            # Get the current file name
            current_files = files[ray_ids[st.session_state.file_index]]

            # Display the current file name

            columns = st.columns(len(current_files))
            # Loop through all DICOM files in the file

            load_image_file = load_dicom_image if file_type=="dcm" else load_image

            for i, file_path in enumerate(current_files):
                with columns[i]:

                    # Load the DICOM image and apply window/level adjustments
                    image_array, default_window_width, default_window_level = load_image_file(file_path)

                    # Create sliders for window width and window level
                    window_width = st.slider(f"Window Width_{i}", 
                                            min_value=1, 
                                            max_value=int(image_array.max()), 
                                            value=int(default_window_width)
                                            )
                    window_level = st.slider(f"Window Level_{i}", 
                                            min_value=int(image_array.min()),
                                            max_value=int(image_array.max()),
                                            value=int(default_window_level)
                                            )

                    # Apply window/level and display the image
                    image_array = apply_window_level(image_array, window_width, window_level)
                    image = Image.fromarray(image_array)
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

            if enter_results:
                save_annotations(csv_path, annotations)
                next_file()
                # MEGA UGLY FIX OF UNCHECKING CHECKBOX
                for subselects in selections[category]:
                    del st.session_state[f"checkbox_{subselects}"]
                    st.session_state[f"checkbox_{subselects}"] = False
                
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
    file_path: str = os.getenv('FILE_PATH', '')
    mode: str = os.getenv("STUDY_MODE", "original")
    file_type: str = os.getenv("FILE_TYPE", "dcm")

    if not file_path:
        st.error("FILE_PATH environment variable must be set.")
    else:
        main(file_path, file_type, mode)