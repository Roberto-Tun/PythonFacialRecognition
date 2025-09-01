FROM python:3.11

WORKDIR /app


RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Start the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
