# Use an official Python runtime as a parent image
FROM python:3.11-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY requirements.txt .
COPY assistant.py .
COPY app.py .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit will run on. App Hosting maps 8080 by default.
EXPOSE 8080

# Command to run the Streamlit app when the container starts
# --server.port 8080: tells Streamlit to bind to this port
# --server.enableCORS false and --server.enableXsrfProtection false:
#   These are often needed for deployment, but be aware of security implications for production apps.
#   Consider setting these based on your specific security needs and domain configuration.
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port", "8080", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]
