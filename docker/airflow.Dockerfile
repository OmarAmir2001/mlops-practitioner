FROM apache/airflow:3.3.0
USER root
COPY  pyproject.toml /opt/prodml/pyproject.toml
COPY src/ /opt/prodml/src/
RUN chown -R airflow: /opt/prodml
USER airflow
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    /opt/prodml