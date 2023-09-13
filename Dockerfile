FROM python:3.8
RUN mkdir /app
WORKDIR /app
ADD . /app/
RUN pip install -r requirements.txt
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port","8080", "--reload", "--reload-dir", "app"]
EXPOSE 8080
