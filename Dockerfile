FROM python:3.11-slim

WORKDIR /app

RUN pip install poetry && \
    poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock README.md ./
RUN poetry install --only main --no-root --no-interaction --no-ansi

COPY src/ ./src/
RUN poetry install --only main --no-interaction --no-ansi

ENTRYPOINT ["piqc"]
CMD ["scan", "--mode", "incluster", "--format", "table"]
