FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# 시스템 패키지 (opencv headless라도 libgl 필요할 때 있음)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    zip \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# 작업 경로
WORKDIR /app/5th_submission

# 파이썬 의존성 설치 (캐시 효율 위해 먼저 복사)
COPY env/requirements.txt /app/5th_submission/env/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r env/requirements.txt

# 프로젝트 복사
COPY . /app/5th_submission

# 런타임 디렉토리 준비 (절대경로 /content)
RUN mkdir -p /content && chmod -R 777 /content

# 환경변수: 기본 런타임 경로
ENV RUNTIME_ROOT=/content
ENV PYTHONPATH=/app/5th_submission
ENV PYTHONUNBUFFERED=1

# 기본 실행 커맨드(선택): 채점 스크립트가 별도로 커맨드 주면 무시됨
CMD ["python", "inference.py"]