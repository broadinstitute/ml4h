FROM gcr.io/broad-ml4cvd/deeplearning:tf2.19-latest-cpu

# Set the working directory
WORKDIR /app

# Install TensorFlow (or any other necessary libraries)
RUN pip install tensorflow

# Copy the Keras model file into the Docker image
COPY ecg_5000_hf_quintuplet_dropout_v2023_04_17.keras /app/ecg_5000_hf_quintuplet_dropout_v2023_04_17.keras

# Copy the Python script
COPY process_files.py /app/process_files.py

RUN pip install ml4h

# Define the command to run the script
CMD ["python", "process_files.py", "/data"]