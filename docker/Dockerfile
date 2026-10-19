FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /bin/uv
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HOME=/cache/huggingface \
    UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy PATH=/opt/venv/bin:$PATH
# The repository is mounted at /work at runtime (docker/compose.yaml): the image holds only the locked dependencies
# and an editable install of tacos pointing there, so code changes need no rebuild.
WORKDIR /work
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-install-project --extra dev
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=.,target=/work,rw \
    uv sync --frozen --extra dev
CMD ["bash"]
