import os
from pathlib import Path
from glob import glob
import json
from typing import List, Dict, Tuple
from datetime import datetime
import streamlit as st
import streamlit.components.v1 as components
import pydicom
import numpy as np
import pandas as pd
from PIL import Image
import yaml


def apply_window_level(
    dicom_array: np.ndarray, window_width: float, window_level: float, invert=False
) -> np.ndarray:
    """Applies window leveling to a DICOM image array.

    Args:
        dicom_array (np.ndarray): The DICOM image pixel array.
        window_width (float): The width of the window.
        window_level (float): The level of the window.

    Returns:
        np.ndarray: The window-leveled image array as an 8-bit unsigned
        integer.
    """
    lower_bound: float = window_level - (window_width / 2)
    upper_bound: float = window_level + (window_width / 2)

    # Clip and scale the pixel array to the [0, 255] range.
    dicom_array = np.clip(dicom_array, lower_bound, upper_bound)
    dicom_array = (dicom_array - lower_bound) / (upper_bound - lower_bound)
    if invert:
        dicom_array = 1 - dicom_array
    dicom_array = dicom_array * 255.0
    return dicom_array.astype(np.uint8)


def load_dicom_image(dicom_path: str) -> tuple[np.ndarray, float, float]:
    """Loads a DICOM file and extracts the pixel array along with window width
    and level.

    Args:
        dicom_path (str): Path to the DICOM file.

    Returns:
        tuple: A tuple containing the DICOM image array, window width, and
        window level.
    """
    dicom = pydicom.dcmread(dicom_path)
    dicom_array = dicom.pixel_array
    window_width: float = (
        dicom_array.max() - dicom_array.min()
    )  # dicom.WindowWidth if 'WindowWidth' in dicom else
    window_level: float = (
        dicom_array.max() + dicom_array.min()
    ) / 2  # dicom.WindowLevel if 'WindowLevel' in dicom else
    return dicom_array, window_width, window_level


def load_image(image_path: str) -> tuple[np.ndarray, float, float]:
    """
    Loads a image (e.g., PNG, JPEG) and extracts the pixel array along with a
    calculated window width and level.

    Args:
        image_path (str): Path to the image file.

    Returns:
        tuple: A tuple containing the image array, window width, and window
        level.
    """
    # Load image using Pillow and convert it to grayscale
    image = Image.open(image_path).convert("L")

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
    return dataframe.to_csv(index=False).encode("utf-8")


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

        keys = list(set(df.keys()) - {"ID"})
        indices = orig_df.ID.astype(int) == df.iloc[0].ID.astype(int)

        orig_df.loc[indices, keys] = df.loc[[0], keys].values

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
        with open(str(csv_path), "w+") as results_file:
            header = (
                "ID, annotated,text_comment, time, ".replace(" ", "")
                + ",".join(
                    [
                        ",".join([f"{elem},certainty_{elem}" for elem in selects[key]])
                        for key in selects.keys()
                    ]
                )
                + "\n"
            )
            results_file.write(header)

            row_template = ", -1," + ",".join(
                [
                    ",".join(["False,-1" for elem in selects[key]])
                    for key in selects.keys()
                ]
            )
            for id in ids:
                results_file.writelines(f"{id},{False},{row_template}\n")


def get_all_files(
    file_path: str, prefix: str, file_type: str, mode: str, tmp_storage: str
) -> Tuple[List[int], Dict[int, List[str]]]:
    """
    Retrieves all files based on specified file path, prefix, and file type.

    Args:
        file_path (str): The base directory where files are located.
        prefix (str): The prefix for the files to filter.
        file_type (str): The type of files to search for (e.g., txt, csv).
        mode (str): The file mode or file identifier (can be multiple,
                    separated by '|').
        tmp_storage (str): The path to a temporary storage file.

    Returns:
        Tuple[List[int], Dict[int, List[str]]]: A tuple containing:
            - List of ray_ids (int): Unique IDs extracted from filenames.
            - Dictionary of all files where the key is the ray ID and the
                value is a list of file paths.
    """

    # If the temporary storage file does not exist, fetch files from the
    # directory
    if not os.path.isfile(tmp_storage):
        # Get a list of all files matching the pattern in the given directory
        ray_files = glob(str(Path(file_path) / "*"))

        # Extract unique IDs (numbers after 'ray' prefix) from the filenames
        ray_ids = list(
            {
                int(
                    Path(file).stem.split("_")[0][3:]
                )  # Extract the numeric part after 'ray'
                for file in ray_files
            }
        )

        # Create a dictionary where each key is a ray_id and the value is a
        # list of matching files
        all_files = {
            index: sum(
                [
                    glob(str(Path(file_path) / f"{prefix}{index:03}" / f"*_{st}.*"))
                    for st in mode.split("|")
                ],
                [],
            )
            for index in ray_ids
        }

        # Fix for Matthias
        print(all_files)
        if file_type != "png":
            all_files = {
                index: [
                    i
                    for i in all_files[index]
                    if ((file_type in i) and ("rayvolve" not in i))
                    or (("png" in i) and ("rayvolve" in i))
                ]
                for index in ray_ids
            }

        # import pdb;pdb.set_trace()

        with open(tmp_storage, "w") as f:
            json.dump(all_files, f)

    else:
        # Load file data from the tmp_storage JSON file
        with open(tmp_storage, "r") as f:
            all_files = json.load(f)

        all_files = {int(k): v for k, v in all_files.items()}
        # Convert the dictionary keys (ray_ids) to integers
        ray_ids = all_files.keys()

    # Return the list of ray_ids and the dictionary of all files
    return ray_ids, all_files


def load_fields(category, selections, descriptor_yml, csv_path):

    current_annotations = pd.read_csv(csv_path)

    conditions: Dict[str, bool] = {}
    for subselect in selections[category]:
        conditions[subselect] = st.checkbox(
            f"{subselect}",
            key=f"checkbox_{subselect}_{st.session_state.file_index}",
            value=current_annotations[subselect]
            .iloc[st.session_state.file_index]
            .item(),
        )
        if conditions[subselect]:
            conditions[f"certainty_{subselect}"] = st.slider(
                descriptor_yml["certainty"]["certainty_caption"],
                min_value=descriptor_yml["certainty"]["min_certainty"],
                max_value=descriptor_yml["certainty"]["max_certainty"],
                value=(
                    (
                        descriptor_yml["certainty"]["min_certainty"]
                        + descriptor_yml["certainty"]["max_certainty"]
                    )
                    // 2
                    if current_annotations[f"certainty_{subselect}"]
                    .iloc[st.session_state.file_index]
                    .item()
                    == -1
                    else current_annotations[f"certainty_{subselect}"]
                    .iloc[st.session_state.file_index]
                    .item()
                ),
                key=f"slider_{subselect}_{st.session_state.file_index}",
            )

    text_comment: str = st.text_input(
        descriptor_yml["comments"],
        value=(
            current_annotations["text_comment"].iloc[st.session_state.file_index]
            if current_annotations["text_comment"].iloc[st.session_state.file_index]
            == current_annotations["text_comment"].iloc[st.session_state.file_index]
            else ""
        ),
        key=f"comment_{st.session_state.file_index}",
    )

    return conditions, text_comment


def main() -> None:
    """
    Main application logic after login. Loads user configurations, manages
    file navigation, annotations, and saves results.
    """

    if not st.session_state.get("authentication_status"):
        st.switch_page("rayvolve_study_auth.py")

    # Environment variable fallbacks
    users_yml_path: str = os.getenv("USERS_YML", "Users.yml")
    selects_yml_path: str = os.getenv("SELECTS_YML", "select.yml")
    descriptor_yml_path: str = os.getenv("DESCRIPTOR_YML", "descriptor.yml")
    tmp_storage_path: str = os.getenv("TMP_STORAGE", "files.json")
    assignment_csv_path: str = os.getenv("ASSIGNMENT_CSV", "users.csv")
    file_path: str = os.getenv("FILE_PATH", "")
    mode: str = os.getenv("STUDY_MODE", "original")
    run_mode: str = os.getenv("RUN_MODE", "original")
    file_type: str = os.getenv("FILE_TYPE", "dcm")

    # Load configuration files
    with open(users_yml_path, "r") as file:
        config: Dict = yaml.safe_load(file)
    credentials: Dict = config["credentials"]

    with open(descriptor_yml_path, "r") as file:
        descriptor_yml: Dict = yaml.safe_load(file)

    with open(selects_yml_path, "r") as file:
        selections: Dict = yaml.safe_load(file)

    authenticator = st.session_state["authenticator"]

    # Set up the logout button
    if st.session_state.get("authentication_status"):
        authenticator.logout(descriptor_yml["login"]["logout"], "sidebar")

    # Get user details from session state
    name: str = st.session_state["name"]
    username: str = st.session_state["username"]

    if config.get("case_assignment", False) and config.get(
        "case_assignment", False
    ).get("external_file", False):
        elem = pd.read_csv(assignment_csv_path, delimiter=";")
        if len(elem[elem.reader == username]) > 0:
            ids: list[int] = [
                int(i[3:])
                for i in elem[elem.reader == username].iloc[0].cases.split(",")
            ]
        else:
            ids = []
    else:
        ids: list[int] = list(
            range(
                credentials["usernames"][name]["startid"],
                credentials["usernames"][name]["endid"],
            )
        )

    prefix: str = descriptor_yml["study_prefix"]

    # Get all files and filter based on user-specific start and end IDs
    ray_ids, all_files = get_all_files(
        file_path, prefix, file_type, mode, tmp_storage_path
    )

    ray_ids = [index for index in ids if index in ray_ids]
    all_files = {index: all_files[index] for index in ray_ids}

    # Map session states to IDs
    session_states: Dict[int, int] = {ray_id: i for i, ray_id in enumerate(ray_ids)}
    files: Dict[int, List[str]] = {index: all_files[index] for index in ray_ids}

    # Define the path to save the results CSV file
    csv_path: Path = Path(__file__).parent / "results" / f"{username}-results.csv"

    # Write results to CSV file
    write_csv(csv_path, selections, ray_ids)

    # Load the CSV file and ensure the 'annotated' column is boolean
    df: pd.DataFrame = pd.read_csv(str(csv_path))
    df["annotated"] = df["annotated"].astype(bool)

    done: bool = len(files) == 0

    # Get the first unannotated file or mark as done
    try:
        try:
            df[~(df["annotated"])].ID.tolist()[0]
        except IndexError:
            done = True
            st.write("No more files! You are done!")
            st.balloons()
    except AttributeError:
        raise AttributeError("Data directory does not contain any images")

    if not done:
        # Initialize session state for file index if not set
        if "file_index" not in st.session_state:
            st.session_state.file_index = 0

        # Define handlers for navigation buttons (next and previous)
        def next_file() -> None:
            if st.session_state.file_index < len(files) - 1:
                st.session_state.file_index += 1
            components.html(
                """
                <script>
                    window.parent.document.querySelector('section.main').scrollTo(0, 0);
                </script>
                """,
                height=0,
            )
            # reset timer
            st.session_state.current_time: float = datetime.now()

        def prev_file() -> None:
            if st.session_state.file_index > 0:
                st.session_state.file_index -= 1
            components.html(
                """
                <script>
                    window.parent.document.querySelector('section.main').scrollTo(0, 0);
                </script>
                """,
                height=0,
            )
            # reset timer
            st.session_state.current_time: float = datetime.now()

        # Display navigation buttons
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            st.button(descriptor_yml["selection"]["previous"], on_click=prev_file)
        with col2:
            with st.expander(descriptor_yml["selection"]["missing"]):
                max_show_files: int = descriptor_yml["selection"]["max_missing"]
                missing_ids: List[int] = df[~df.annotated].ID.tolist()[:max_show_files]
                missing_cols = st.columns(max_show_files)
                skipping: Dict[int, bool] = {}
                for i, missing_id in enumerate(missing_ids):
                    with missing_cols[i]:
                        skipping[missing_id] = st.button(
                            f"{session_states[missing_id] + 1}"
                        )
                        if skipping[missing_id]:
                            st.session_state.file_index = session_states[missing_id]
        with col3:
            st.button(descriptor_yml["selection"]["next"], on_click=next_file)

        # Display progress bar for annotated files
        progress = int((df["annotated"].sum() / len(files)) * 100)
        st.progress(progress, text=f"{progress} %")

        # Display task information
        with st.expander(descriptor_yml["task"]["task_name"], expanded=True):
            info_md: str = descriptor_yml["task"]["task_caption"]
            st.markdown(info_md, unsafe_allow_html=True)

        # Check if the current file is annotated
        is_annotated: bool = df[
            df.ID == ray_ids[st.session_state.file_index]
        ].annotated.iloc[0]

        if "current_time" not in st.session_state:
            # reset timer
            st.session_state.current_time: float = datetime.now()
        st.session_state.entry_time: float = float(
            df[df.ID == ray_ids[st.session_state.file_index]].time.iloc[0]
        )

        # Display the header with the case and current file index
        if run_mode == "debug":
            st.header(
                f'{descriptor_yml["case"]} {st.session_state.file_index + 1}/{len(files)} - {ray_ids[st.session_state.file_index]} - {"☑" if is_annotated else "☐"}'
            )
        else:
            st.header(
                f'{descriptor_yml["case"]} {st.session_state.file_index + 1}/{len(files)} - {"☑" if is_annotated else "☐"}'
            )

        # Get and display the current file's images
        current_files: List[str] = files[ray_ids[st.session_state.file_index]]
        columns = st.columns(len(current_files))
        load_image_file = load_dicom_image if (file_type == "dcm") else load_image

        for i, file_path in enumerate(current_files):
            with columns[i]:

                # fix for matthias
                if "rayvolve" in file_path:
                    image_array, default_window_width, default_window_level = (
                        load_image(file_path)
                    )
                else:
                    image_array, default_window_width, default_window_level = (
                        load_image_file(file_path)
                    )

                if (
                    f"inversion_{i}_{st.session_state.file_index}"
                    not in st.session_state.keys()
                ):
                    st.session_state[f"inversion_{i}_{st.session_state.file_index}"] = (
                        False
                    )
                if st.button(
                    "Invert Image",
                    key=f"button_inversion_{i}_{st.session_state.file_index}",
                ):
                    st.session_state[f"inversion_{i}_{st.session_state.file_index}"] = (
                        not st.session_state[
                            f"inversion_{i}_{st.session_state.file_index}"
                        ]
                    )
                window_width = st.slider(
                    f"Window Width_{i}",
                    min_value=1,
                    max_value=int(image_array.max()),
                    value=int(default_window_width),
                )
                window_level = st.slider(
                    f"Window Level_{i}",
                    min_value=int(image_array.min()),
                    max_value=int(image_array.max()),
                    value=int(default_window_level),
                )
                image_array = apply_window_level(
                    image_array,
                    window_width,
                    window_level,
                    st.session_state[f"inversion_{i}_{st.session_state.file_index}"],
                )
                image = Image.fromarray(image_array)
                st.image(image, use_column_width=True)

        # Handle annotations
        category = Path(current_files[0]).stem.split("_")[1]
        conditions, text_comment = load_fields(
            category, selections, descriptor_yml, csv_path
        )

        # Store annotations
        annotations: List[Dict] = [
            {
                "ID": ray_ids[st.session_state.file_index],
                "text_comment": text_comment,
                "annotated": True,
                **conditions,
            }
        ]

        # Submit button for results
        enter_results = st.button(descriptor_yml["submit"])
        if enter_results:

            annotations[0]["time"] = float(
                st.session_state.entry_time
                + (datetime.now() - st.session_state.current_time).total_seconds()
            )
            save_annotations(csv_path, annotations)

            next_file()

            # reset timer
            st.session_state.current_time: float = datetime.now()

            # Reset checkboxes
            # for subselect in selections[category]:
            #     del st.session_state[f"checkbox_{subselect}"]
            #     st.session_state[f"checkbox_{subselect}"] = False

            st.rerun()

    # Download results button
    with open(csv_path, "rb") as f:
        st.download_button(
            label=descriptor_yml["download"],
            data=f,
            file_name=f"{username}-results.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
