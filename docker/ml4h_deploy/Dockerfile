FROM ghcr.io/broadinstitute/ml4h:tf2.9-latest-cpu

# Set the working directory
WORKDIR /app

# Install TensorFlow (or any other necessary libraries)
RUN pip install tensorflow

# Copy the Keras model file into the Docker image
COPY ecg2af_quintuplet_v2024_01_13.h5 /app/ecg2af_quintuplet_v2024_01_13.h5

# Copy the Python script
COPY process_files.py /app/process_files.py

RUN pip3 install ml4h

# Define the command to run the script
CMD ["python", "process_files.py", "/data"]