# 🧩 Industrial Multi-Scenario Forecasting (ShujingCup 2025 1st Prize Solution)

> 🌟 工业多场景时间序列预测框架（基于 DLinear）  
> 🏆 第八届“数境杯”数据智能创新应用大赛 · 大模型工业多重场景挑战赛 **开源复现**  
> 📦 完整可复现版本（含代码、模型、Docker 环境、推理脚本）

---

## 📌 项目简介

本项目为 2025 数境杯工业多场景挑战赛 **三等奖** 获奖方案的开源实现，聚焦三种截然不同的工业场景：

| 赛题 | 场景 | 预测任务 |
|------|------|----------|
| Task 1 | 风电场 | 短期风速/风功率预测 |
| Task 2 | 水电站 | 长期入库流量预测 |
| Task 3 | 离散制造业 | 稀疏月度物料需求预测 |

与大多数复杂 Transformer / 时序大模型不同，本方案采用：

> **Decomposition-based Linear Time Series Model (DLinear) + Unified Pipeline + Industrial Constraints**

实现了：
- 高精度
- 可解释
- 通用
- 完全复现
- 极低算力需求（全部运行于 Colab/Kaggle 免费 GPU）

---

## 📁 目录结构

```

industrial-multi-scenario-forecasting/
│
├── data/                      # 示例数据（需用户自行放入原始数据）
│── discussion/                # 赛题解析与思路 
├── model/                     # 预训练模型权重
│   ├── last_dlinear_wind.pth
│   ├── last_dlinear_inflow.pth
│   └── last_dlinear_demand.pth
│
├── infer_task1.py             # 赛题1推理脚本
├── infer_task2.py             # 赛题2推理脚本
├── infer_task3.py             # 赛题3推理脚本
│
├── requirements.txt           # 依赖列表
├── Dockerfile                 # 复现实验镜像构建文件
└── README.md                  # 当前文档

````

---

## ⚙️ 环境与依赖

| 组件 | 版本 |
|------|------|
| Python | 3.10 |
| PyTorch | 2.1+ |
| NumPy | 1.26+ |
| Pandas | 2.2+ |
| Scikit-learn | 1.5+ |
| OS | Debian (python:3.10-slim) |
| GPU | 可选（CPU 即可复现） |

> 💡 所有依赖均通过 `requirements.txt` 安装，无需自行配置 CUDA。

---

## 🚀 快速运行（无需开发环境）

### 1️⃣ 加载镜像

```bash
docker load -i competition_final_v1.tar
````

### 2️⃣ 执行全部推理

```bash
docker run --rm -v $(pwd)/data:/app/data competition_final:v1
```

镜像将执行：

```
python infer_task1.py &&
python infer_task2.py &&
python infer_task3.py
```

### 3️⃣ 输出结果

生成：

```
submission_task1.csv
submission_task2.csv
submission_task3.csv
```

---

## 🧠 单独运行任务（可选）

```bash
docker run --rm -v $(pwd)/data:/app/data competition_final:v1 python infer_task3.py
```

---

## 📦 镜像构建说明（如需重新训练或开发）

```bash
docker build -t competition_final:v1 .
```

---

## 🔬 核心技术亮点

### ⭐ 多场景统一数据范式

* 使用滑动窗口策略适配不同时间尺度
* 自动时间序列展开，适配短期/长期/稀疏数据
* 完全不依赖手动特征工程

### ⭐ 模型方法：Decomposition-based Linear (DLinear)

* 序列分解：趋势 + 季节 + 残差
* 线性结构替代复杂注意力
* 小数据、工业环境下稳定优于 Transformer

### ⭐ 工业可复现设计

* 100% 无外部依赖
* 完全 Docker 化
* CPU 可复现
* 单文件可执行脚本（无需 Notebook）

---

## 📊 各赛题技术细节

| 赛题   | 预测脚本           | 模型                      | 输出                   |
| ---- | -------------- | ----------------------- | -------------------- |
| 风电预测 | infer_task1.py | last_dlinear_wind.pth   | submission_task1.csv |
| 入库流量 | infer_task2.py | last_dlinear_inflow.pth | submission_task2.csv |
| 物料需求 | infer_task3.py | last_dlinear_demand.pth | submission_task3.csv |

---

## 📌 未来改进方向

* 集成 Diffusion/TS-GAN 合成工业数据
* Neural ODE for irregular timestamps
* DLinear + LLM adapter for hybrid TS+NLP
* Multitask shared parameter training

---


