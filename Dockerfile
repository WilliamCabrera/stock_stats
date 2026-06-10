# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# System deps for numpy/pandas/vectorbt compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Optional: install TA-Lib C library if you want the TA-Lib Python package
# RUN apt-get install -y --no-install-recommends wget && \
#     wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
#     tar -xzf ta-lib-0.4.0-src.tar.gz && cd ta-lib && \
#     ./configure --prefix=/usr && make && make install

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .

EXPOSE 8000
