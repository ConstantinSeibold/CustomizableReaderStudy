# Rayvolve Study Authentication Application

This project is a Streamlit-based web application designed for authenticated annotation of medical studies. The application allows users to log in, view DICOM images, and annotate them with specific conditions, ensuring that results are securely saved and traceable.

## Features

- **User Authentication**: Secured login using credentials defined in a YAML file (`Users.yml`).
- **DICOM Image Handling**: Load, display, and manipulate DICOM images with windowing features.
- **Annotation Workflow**: Guided annotation process with options for multiple conditions and certainty levels.
- **CSV Export**: Annotations are saved locally and can be downloaded as a CSV file.

## Project Structure

```
.
├── rayvolve_study_auth.py   # Main script to start the Streamlit app
├── hash_password.py         # Utility for managing hashed passwords
├── pages/
│   └── study.py             # Study-specific pages

```

## Prerequisites

- Python 3.8 or later
- Docker (optional, for containerized deployment)
- The following Python packages:
  - `streamlit`
  - `pydicom`
  - `numpy`
  - `pandas`
  - `Pillow`
  - `yaml`
  - `streamlit_authenticator`


### Docker Installation Guide

If Docker is not installed on your system, follow the [official Docker installation guide](https://docs.docker.com/engine/install/) to get started.


## ⚙️ Environment Variables Explained

The app relies on several environment variables for configuration. These variables can be passed either through the Docker command or via environment files. Here's what they mean:

| Variable |	Default | Value	Description |
| ----------- | ----------- | ----------- | 
USERS_YML	| Users.yml |	Path to the YAML file containing user credentials and session configurations.
ASSIGNMENT_CSV	| users.csv |	Path to a CSV file mapping users to assigned cases.
SELECTS_YML	| select.yml	| Path to the YAML file defining selectable study elements.
DESCRIPTOR_YML	| descriptor.yml |	Path to the YAML file with UI text and study descriptors.
TMP_STORAGE	| files.json	| Temporary storage file for session data.
FILE_PATH	| '' |	Path to the directory containing directories with study files (e.g., DICOM images).
 | |_ | Directory structure {3 letter abbreviation for project name}{case Number}/{3 letter abbreviation for project name}{case Number}_{Case name shown in select.yml}_{F for frontal, L for Lateral}_{STUDY_MODE}.{FILE_TYPE}
STUDY_MODE |	original	| Study mode identifier, such as rayvolve or original. Expects files to have filename structure of .
RUN_MODE |	original	| App runtime mode. ```original``` for the straight app. ```debug``` for debug purposes to display real file names. 
FILE_TYPE |	dcm |	Type of files used in the study (e.g., ```dcm``` for DICOM, otherwise expects jpg).


## 🚀 How to Run the App

### Option 1: Using Streamlit

Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

To run the app locally using Streamlit, execute the following command:

```bash
streamlit run rayvolve_study_auth.py
```

### Option 2: Using Docker Locally



```bash
docker build -t rayvolve .

docker run -d -p 8501:8501 \
  -v /path/to/Users.yml:/app/Users.yml \
  -v /path/to/select.yml:/app/select.yml \
  -v /path/to/study_samples/:/app/study_samples \
  -v /path/to/descriptor.yml:/app/descriptor.yml \
  -v /path/to/users.csv:/app/users.csv \
  -v /path/to/results:/app/pages/results/ \
  -e USERS_YML=/app/Users.yml \
  -e SELECTS_YML=/app/select.yml \
  -e DESCRIPTOR_YML=/app/descriptor.yml \
  -e FILE_TYPE=dcm \
  -e FILE_PATH=/app/study_samples \
  -e STUDY_MODE=rayvolve \
  -e ASSIGNMENT_CSV=/app/users.csv \
  rayvolve
```

### Option 3: Using Docker from DockerHub

You can also run the app using Docker. Use the following command:

```bash
docker run -d -p 8501:8501 \
  -v /path/to/Users.yml:/app/Users.yml \
  -v /path/to/select.yml:/app/select.yml \
  -v /path/to/study_samples/:/app/study_samples \
  -v /path/to/descriptor.yml:/app/descriptor.yml \
  -v /path/to/users.csv:/app/users.csv \
  -v /path/to/results:/app/pages/results/ \
  -e USERS_YML=/app/Users.yml \
  -e SELECTS_YML=/app/select.yml \
  -e DESCRIPTOR_YML=/app/descriptor.yml \
  -e FILE_TYPE=dcm \
  -e FILE_PATH=/app/study_samples \
  -e STUDY_MODE=rayvolve \
  -e ASSIGNMENT_CSV=/app/users.csv \
  cseibold/rayvolve:amd.0.9
```

## Configuration Files

### Users.yml

Defines user credentials, session management, and case assignments (`startid`,`endid`). Passwords can be in cleartext or hashed. The container will hash cleartext passwords upon startup.

Example Configuration:
```yml
case_assignment:
  external_file: true # If you are loading case assignments via an external csv file. Otherwise on false to set start_id and end_id
cookie:
  expiry_days: 0
  key: random_signature_key
  name: random_cookie_name
credentials:
  usernames:
    USER1: # name required for external csv and under which results are stored
      email: example@mail.com
      failed_login_attempts: 0
      logged_in: false
      name: John Doe # Use this name in the login window
      password: $2b$12$examplehashedpassword

```

### select.yml

Defines selectable elements for study annotations with the following structure:

```yml
Case_Type:
  - Option 1
  - Option 2
  - Option 3
```

Example for Rayvolve:

```yaml
RK:
  - Olecranon
  - Radiusköpfchen
  - Epicondylus medialis
  - Epicondylus lateralis
  - Proc. coronoideus
  - Humerusschaft
  - Ulnaschaft
  - Radiusschaft

SCP:
  - Radiusfraktur
  - Proc. styloideus ulnae
  - Scaphoid
  - Andere Handwurzelknochen
  - Mittelhandknochen
  - Phalanx proximalis/media/distalis
```

### descriptor.yml

Defines UI text, captions, and study-specific settings. Example:

```yml
study_prefix: "RAY"
login: # Field captions for Login mask
  study_name: "Rayvolve - Nutzerstudie"
  username: "Nutzername"
  password: "Password"
  login: "Login"
  logout: "Logout"
  error: "Username/password is incorrect"
  warning: "Please enter your username and password"
task: # Field captions for Task descriptions
  task_name: "Aufgabenstellung" 
  task_caption: |
    ### Bitte klicken Sie alles an was zutrifft.
    ### Daraufhin erscheint ein Slider. Geben Sie bitte durch diesen Slider an, wie sicher Sie sich dabei sind.
    ### Nachdem Sie Ihre Bewertung angegeben haben, klicken Sie bitte auf 'Submit'.
case: "Case"
selection: # Field captions for Page management
  previous: "Previous"
  next: "Next"
  missing: "Missing:"
  max_missing: 10
certainty:  # Define certainties along the slider, default is center
  min_certainty: 1 
  max_certainty: 5 
  certainty_caption: "Sicherheit: 1 (sehr unsicher) - 5 (sehr sicher) "
comments: "Kommentarfeld"
submit: "Abschicken"
download: "Download Annotation CSV"
```

### Users.csv

External file which maps readers to specific cases. Relies on ```external_file: true``` in the ```Users.yml``` Example:

```csv
reader;count;cases
user1;76;RAY014,RAY001,RAY022,RAY993
```

## 📂 Study Directory Structure

Ensure the following directory and file structures are in place for the study:

```
/path/to/study_samples/       # Directory containing study files (e.g., DICOM images).
/path/to/results/             # Directory for saving results.
Users.yml                     # User credentials and settings.
select.yml                    # Selectable elements for annotations.
descriptor.yml                # UI and text descriptions.
users.csv                     # Reader-to-case assignments.

```
