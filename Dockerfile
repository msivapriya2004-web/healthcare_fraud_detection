FROM python:3.9-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files (including the trained_model.joblib generated in CI)
COPY . .

EXPOSE 8000

# Start the API directly
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
