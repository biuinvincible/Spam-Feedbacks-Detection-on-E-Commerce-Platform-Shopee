FROM python:3.10

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run preprocessing with explicit path
RUN python src/preprocessing.py

# Since pipeline.py is in the root directory
CMD ["python", "pipeline.py"]