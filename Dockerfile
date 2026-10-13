FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY healthcare_lora ./healthcare_lora
RUN pip install --no-cache-dir .
RUN useradd --create-home researcher && chown -R researcher /app
USER researcher
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "healthcare_lora.api:app", "--host", "0.0.0.0", "--port", "8000"]
