FROM python:3.9

RUN mkdir -p /app

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .
COPY sample_files/configs/sample_configs.json ./app/configs.json

CMD ["sleep", "infinity"]
