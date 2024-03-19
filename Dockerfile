# Use the official Python 3.11 image as the base image
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file to the working directory
COPY requirements.txt .

# Create a virtual environment and activate it
RUN python -m venv venv
RUN . venv/bin/activate

# Install the Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything over
COPY . .

# Run the model_eval.py script with the specified argument
CMD ["python3", "model_eval.py", "CodeParrotSmall_agpl3_python_2023-03-27-21-21-29"]