# ======================
# 基础镜像（使用本地已下载的官方 python 镜像）
# ======================
FROM python:3.10-slim

# ======================
# 工作目录设置
# ======================
WORKDIR /app

# ======================
# 拷贝项目文件
# ======================
COPY . /app

# ======================
# 安装国内 pip 源并安装依赖
# ======================
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install --no-cache-dir -r requirements.txt

# ======================
# 设置默认命令
# 一次性执行三个赛题推理脚本
# ======================
CMD ["sh", "-c", "python infer_task1.py && python infer_task2.py && python infer_task3.py"]
