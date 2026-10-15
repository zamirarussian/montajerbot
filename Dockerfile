FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Inter shrifti (SIL OFL) build vaqtida yuklanadi
RUN mkdir -p fonts && curl -sSL -o /tmp/inter.zip https://github.com/rsms/inter/releases/download/v4.0/Inter-4.0.zip \
    && for f in InterDisplay-Bold InterDisplay-Black Inter-Medium Inter-SemiBold; do unzip -p /tmp/inter.zip extras/otf/$f.otf > fonts/$f.otf; done \
    && rm /tmp/inter.zip
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "app.bot"]
