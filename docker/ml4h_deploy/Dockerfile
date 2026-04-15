FROM ghcr.io/broadinstitute/ml4h:tf2.19-latest-cpu
# Set the working directory
WORKDIR /app

# Install TensorFlow (or any other necessary libraries)
RUN pip install tensorflow

# Copy the Keras model file into the Docker image
COPY ecg_cnn_scratch_26task_v2025_08_11.keras /app/ecg_cnn_scratch_26task_v2025_08_11.keras

# Copy the Python script
COPY process_files.py .

RUN pip install ml4h

# Define the command to run the script
ENTRYPOINT ["python", "process_files.py"]
