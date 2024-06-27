FROM python:3.11

RUN pip install poetry==1.8.3

WORKDIR /app


COPY pyproject.toml poetry.lock ./
RUN poetry install

COPY src ./src

EXPOSE 8000
CMD ["poetry", "run", "uvicorn", "src.server:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-include", "*.pdf"]
# CMD ["poetry", "run", "python", "src/server.py" ]
