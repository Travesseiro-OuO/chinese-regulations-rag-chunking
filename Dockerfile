FROM python:3.11-slim@sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple numpy==2.2.6 onnxruntime==1.22.1 tokenizers==0.21.2 requests==2.32.4 pymupdf==1.26.3 beautifulsoup4==4.13.4
WORKDIR /workspace
USER 10001:10001
CMD ["python", "src/experiment.py"]
