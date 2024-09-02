import streamlit_authenticator as stauth
import yaml
import os
import re

def hide_users():
    # Function to check if a password is already hashed
    def is_hashed(password):
        # This checks if the password starts with the bcrypt prefix
        return re.match(r'^\$2[aby]?\$[0-9]{2}\$[./A-Za-z0-9]{53}$', password) is not None

    users_yml_path = os.getenv('USERS_YML', 'Users.yml')

    # Load your YAML file containing the user credentials
    with open(users_yml_path, "r") as file:
        credentials = yaml.safe_load(file)

    # Extract the usernames and plaintext passwords
    usernames = credentials['credentials']['usernames']

    # Hash the passwords and update the dictionary
    for username, info in usernames.items():
        if not is_hashed(info['password']):
            hashed_password = stauth.Hasher([info['password']]).generate()[0]
            info['password'] = hashed_password

    # Save the updated credentials back to the YAML file
    with open(users_yml_path, "w") as file:
        yaml.dump(credentials, file, default_flow_style=False)

    print("Passwords have been hashed and saved back to the file.")
