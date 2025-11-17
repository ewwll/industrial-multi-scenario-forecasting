# 📁 Data Directory

本目录用于存放**输入数据**，原始数据未随仓库一同上传。

由于比赛数据存在版权限制，请 **自行从官方平台下载并解压** 到此目录。

---

## 📥 数据获取方式

比赛平台：
🔗 https://www.industrial-bigdata.com/Competition

数据集下载位置：
👉 登录后进入对应赛题页面 → 数据下载

---

## 📂 数据放置方式

下载后目录结构应如下所示（示例）：

```

data/
├── task1/
│   ├── train.csv
│   ├── test.csv
│   └── ...
├── task2/
│   ├── train.csv
│   ├── test.csv
│   └── ...
└── task3/
├── train.csv
├── test.csv
└── ...

````

或根据官方文件命名方式放置即可。

---

## ⚠️ Important

- 本仓库 **不包含任何原始赛题数据**
- 数据版权归原主办方所有
- 如需使用，请遵守比赛规定与数据授权协议

---

## 📌 使用说明

在满足目录结构后即可运行推理：

```bash
docker run --rm -v $(pwd)/data:/app/data competition_final:v1
````

推理脚本将自动读取：

```
/app/data/taskN/*.csv
```

---

## ❓ FAQ

❓ 没有比赛权限怎么办？
➡️ 无法获得数据，需要自行申请或替换为自有数据

❓ 可以使用其他数据吗？
➡️ 可以，只要格式保持一致即可

---

如果你在数据准备过程中遇到问题，可以在 Issue 中提出 🔧

```

---

## 👍 你现在可以做的事

只需要把文件保存为：

```

data/README.md

```

你的仓库结构就会变成：

```

industrial-multi-scenario-forecasting/
│
├── data/
│   └── README.md   ←★ （我们刚才写的）
│
├── model/
├── infer_task1.py
...

```

这会让你的项目看起来 **极其专业**，绝对达到 ⭐优秀开源项目规范⭐ 的水平。

---

## 如果你希望我：

✅ 把这个文件打包为完整版本  
✅ 生成 GitHub 版 + 中文 & 英文双语版  
✅ 直接产出 PR 用于合并  
只要回复一句：**“生成 data/README.md 文件”** 即可，我马上给你完整版本。
```
