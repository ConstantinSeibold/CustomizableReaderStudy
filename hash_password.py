import os
import re
import yaml
import streamlit_authenticator as stauth

def hide_users() -> None:
    """Hashes plaintext passwords in the user credentials YAML file if they are not already hashed.

    This function checks each user's password in the credentials file. If a password is not hashed,
    it hashes the password using bcrypt and updates the credentials file with the hashed password.
    """

    def is_hashed(password: str) -> bool:
        """Checks if a password is already hashed using bcrypt.

        Args:
            password (str): The password to check.

        Returns:
            bool: True if the password is hashed, False otherwise.
        """
        # Regex pattern to match bcrypt hashed passwords.
        return re.match(r'^\$2[aby]?\$[0-9]{2}\$[./A-Za-z0-9]{53}$', password) is not None

    # Load the path to the users.yml file from the environment variable or use the default.
    users_yml_path: str = os.getenv('USERS_YML', 'Users.yml')

    # Load the user credentials from the YAML file.
    with open(users_yml_path, "r") as file:
        credentials: dict = yaml.safe_load(file)

    # Extract the usernames and their corresponding information.
    usernames: dict = credentials['credentials']['usernames']

    # Iterate over each user to check and hash their password if necessary.
    for username, info in usernames.items():
        if not is_hashed(info['password']):
            # Hash the password and update the credentials.
            hashed_password: str = stauth.Hasher([info['password']]).generate()[0]
            info['password'] = hashed_password

    # Save the updated credentials back to the YAML file.
    with open(users_yml_path, "w") as file:
        yaml.dump(credentials, file, default_flow_style=False)

    print("Passwords have been hashed and saved back to the file.")
