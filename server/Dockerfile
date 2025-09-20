<!-- =============================
= FILE: server/Dockerfile
============================== -->
# 簡單 Docker 部署：
#   docker build -t motion-uploader .
#   docker run -it --rm -p 8787:8787 --env-file .env -v $PWD:/app motion-uploader
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8787
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8787"]

