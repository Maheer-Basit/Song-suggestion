FROM python:3.10-slim

WORKDIR /code

# Install data science and web server dependencies
RUN pip install fastapi uvicorn pydantic scikit-learn joblib pandas

# Copy the app and the trained AI brain into the container
COPY ./app.py /code/app.py
COPY ./spotify_model.pkl /code/spotify_model.pkl

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]