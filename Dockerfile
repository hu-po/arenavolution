# Use the specified base image
FROM nvcr.io/nvidia/jax:23.10-py3

# Set the working directory in the container
WORKDIR /app

# Copy the Python script into the container
COPY minimamba.py .

# Command to run the Python script
CMD ["python", "./minimamba.py"]