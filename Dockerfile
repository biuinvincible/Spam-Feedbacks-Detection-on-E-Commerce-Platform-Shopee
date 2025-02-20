# Sử dụng Python 3.9 làm base image
FROM python:3.9

# Cài đặt Java (cần cho VnCoreNLP)
RUN apt-get update && apt-get install -y default-jdk && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Thiết lập biến môi trường JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${JAVA_HOME}/bin:${PATH}"
ENV JVM_PATH="${JAVA_HOME}/lib/server/libjvm.so"

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Copy file requirements trước để tận dụng caching layer
COPY requirements.txt /app/

# Cài đặt các dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn vào container
COPY . /app

# Tải trước dữ liệu của nltk để tránh tải online mỗi lần chạy container
RUN python -c "import nltk; nltk.download('punkt_tab')"

# Chạy preprocessing.py trước, sau đó chạy pipeline.py
CMD python src/preprocessing.py && python pipeline.py
