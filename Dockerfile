FROM python:3.12.4-bookworm

# Set the working directory inside the container
WORKDIR /app

COPY requirements.txt /app/
COPY hash_password.py /app/
COPY rayvolve_study_auth.py /app/
COPY Dockerfile /app/

RUN apt update && apt install -y libgl1-mesa-glx
RUN python3 -m pip install --upgrade pip
RUN pip install -r ./requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "rayvolve_study_auth.py","--server.port", "8501"]