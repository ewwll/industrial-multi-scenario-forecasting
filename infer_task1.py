import pandas as pd
df1=pd.read_csv('./data/1/223145164/训练集/风场1/m01/2018-01-01.csv')
df2=pd.read_csv('./data/1/mn225691/filtered_1.csv')
print(df1.head())
print(df2.head())

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# ========== 配置 ==========
root_dir = "./data/1/223145164/训练集"
SEQ_IN, SEQ_OUT = 120, 20
BATCH_SIZE = 64
SAMPLE_PER_TURBINE = 2000  # 每个风机抽样的数量

# ========== 数据集类 ==========
class TurbineDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# ========== 数据处理函数 ==========
def create_sequences(turbine_data, seq_in=SEQ_IN, seq_out=SEQ_OUT):
    """
    turbine_data: list of np.array，每个元素 shape=(2880, 2)
    拼接后 shape=(T, 2)，再切分成滑动窗口
    """
    data = np.concatenate(turbine_data, axis=0)  # shape=(T, 2)
    T, F = data.shape

    X, Y = [], []
    for i in range(T - seq_in - seq_out + 1):
        X.append(data[i:i+seq_in])
        Y.append(data[i+seq_in:i+seq_in+seq_out])

    return np.array(X), np.array(Y)

def process_file(file_path):
    """单文件清洗（带丢弃原因输出）"""
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            #print(f"[丢弃] 空文件: {file_path}")
            return None

        # 解析时间
        df['time'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
        if df['time'].isnull().all():
            print(f"[丢弃] 时间列无法解析: {file_path}")
            return None
        df = df.set_index('time')

        # 取风向和风速
        if not set(['风向','风速']).issubset(df.columns):
            #print(f"[丢弃] 缺少风向或风速: {file_path}")
            return None
        df_values = df[['风向','风速']].astype(float)

        # 丢弃规则
        if df_values.shape[0] != 2880:
            #print(f"[丢弃] 长度不等于2880: {file_path} (实际 {df_values.shape[0]})")
            return None
        if df_values.isnull().any().any():
            #print(f"[丢弃] 含 NaN: {file_path}")
            return None

        return df_values.values  # numpy array
    except Exception as e:
        print(f"[丢弃] 处理失败: {file_path}, 错误: {e}")
        return None
# ========== 主逻辑 ==========
all_X, all_Y = [], []
global_success, global_discard = 0, 0

for farm in os.listdir(root_dir):
    farm_path = os.path.join(root_dir, farm)
    if not os.path.isdir(farm_path):
        continue

    turbines = os.listdir(farm_path)
    print(f"风场 {farm} 下风机数量: {len(turbines)}")

    for turbine in turbines:
        turbine_path = os.path.join(farm_path, turbine)
        if not os.path.isdir(turbine_path):
            continue

        turbine_data = []
        success, discard = 0, 0   # 每个风机的统计

        for file in os.listdir(turbine_path):
            file_path = os.path.join(turbine_path, file)
            if not os.path.isfile(file_path):
                continue
            df_values = process_file(file_path)
            if df_values is None:
                discard += 1
                global_discard += 1
                continue
            turbine_data.append(df_values)
            success += 1
            global_success += 1

        # 打印风机统计
        print(f"风机 {turbine}: 成功 {success} | 丢弃 {discard}")

        if len(turbine_data) == 0:
            continue

        # 生成序列
        X, Y = create_sequences(turbine_data, SEQ_IN, SEQ_OUT)

        # 每个风机只保留 SAMPLE_PER_TURBINE 条
        if len(X) > SAMPLE_PER_TURBINE:
            idx = np.random.choice(len(X), SAMPLE_PER_TURBINE, replace=False)
            X, Y = X[idx], Y[idx]

        all_X.append(X)
        all_Y.append(Y)
        print(f"风机 {turbine}: 抽样后 {len(X)} 条")

# 拼接所有风机
if len(all_X) > 0:
    all_X = np.concatenate(all_X, axis=0)
    all_Y = np.concatenate(all_Y, axis=0)
    print(f"最终训练集: X={all_X.shape}, Y={all_Y.shape}")
else:
    print("没有可用数据！")

# 构建 DataLoader
train_loader = DataLoader(TurbineDataset(all_X, all_Y),
                          batch_size=BATCH_SIZE, shuffle=True)

print(f"DataLoader 就绪，batch 数量: {len(train_loader)}")

# 全局统计
print(f"全局统计: 成功 {global_success} | 丢弃 {global_discard}")
torch.save((all_X, all_Y), "dataset.pt")
np.savez("dataset.npz", X=all_X, Y=all_Y)
import os
import gc
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)

        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.individual = configs.individual
        self.channels = configs.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)

            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0,2,1) # to [Batch, Output length, Channel]
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# ===================
#  数据集封装
# ===================
class TurbineDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# ===================
#  配置参数
# ===================
class Configs:
    def __init__(self):
        self.seq_len = 120      # 输入序列长度
        self.pred_len = 20     # 预测序列长度
        self.enc_in = 2       # 特征维度（输入通道数，记得和数据匹配！）
        self.individual = False

        self.batch_size = 64
        self.lr = 1e-3
        self.epochs = 100
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

configs = Configs()

# ===================
#  数据划分
# ===================
# 假设 all_X, all_Y 已经准备好（来自你之前的数据清洗 + 抽样 + 拼接步骤）
X_train, X_val, Y_train, Y_val = train_test_split(
    all_X, all_Y, test_size=0.2, random_state=42
)

train_loader = DataLoader(TurbineDataset(X_train, Y_train),
                          batch_size=configs.batch_size, shuffle=True)
val_loader = DataLoader(TurbineDataset(X_val, Y_val),
                        batch_size=configs.batch_size, shuffle=False)

print(f"训练集: {len(train_loader)} batch, 验证集: {len(val_loader)} batch")

# ===================
#  模型 & 优化器
# ===================
model = Model(configs).to(configs.device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=configs.lr)


# ===================
#  训练 + 验证循环
# ===================
best_val_loss = float("inf")

# 训练集 DataLoader 形状检查
for batch_idx, (X, Y) in enumerate(train_loader):
    print(f"Train batch {batch_idx}: X.shape={X.shape}, Y.shape={Y.shape}")
    break  # 只打印第一个 batch

# 验证集 DataLoader 形状检查
for batch_idx, (X, Y) in enumerate(val_loader):
    print(f"Val batch {batch_idx}: X.shape={X.shape}, Y.shape={Y.shape}")
    break  # 只打印第一个 batch


for epoch in range(1, configs.epochs + 1):
    # ---- 训练 ----
    model.train()
    train_loss = 0.0
    for X, Y in train_loader:
        X, Y = X.to(configs.device), Y.to(configs.device)

        optimizer.zero_grad()
        #print(X.shape,Y.shape)
        outputs = model(X)  # [B, pred_len, C]
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X.size(0)

    train_loss /= len(train_loader.dataset)

    # ---- 验证 ----
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X, Y in val_loader:
            X, Y = X.to(configs.device), Y.to(configs.device)
            #print(X.shape,Y.shape)
            outputs = model(X)
            loss = criterion(outputs, Y)
            val_loss += loss.item() * X.size(0)

    val_loss /= len(val_loader.dataset)

    print(f"Epoch [{epoch}/{configs.epochs}] "
          f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # 保存最好模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_dlinear.pth")
        print(">>> 更新最优模型保存")

print("训练完成！")

import pandas as pd
df1=pd.read_csv('./data/1/mn225691/(2)/测试集_复赛/风场1/m01/100.csv')
df2=pd.read_csv('./data/1/mn225691/filtered_1.csv')
print(df1.head())
print(df2.head())
df3=pd.read_csv('./data/1/mn225691/(2)/测试集_复赛/风场2/m26/100.csv')
print(df3.head())
print(df3.shape)
import os

def count_folders(path):
    # 使用列表推导式筛选出文件夹
    folder_list = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return len(folder_list), folder_list

# 示例用法
directory_path = "./data/1/mn225691/(2)/测试集_复赛/风场1"
num_folders, folders = count_folders(directory_path)

print(f"路径 '{directory_path}' 下共有 {num_folders} 个文件夹。")
print("文件夹列表：", folders)
fn=0
print(directory_path)
for i in folders:
    path=os.path.join(directory_path,i)
    n=0
    print(path)
    for j in os.listdir(path):
        n=n+1
    print(f"路径 '{path}' 下共有 {n} 个文件。")
    fn=fn+n
print("文件夹总数：", fn)  
import os

def count_folders(path):
    # 使用列表推导式筛选出文件夹
    folder_list = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return len(folder_list), folder_list

# 示例用法
directory_path = "./data/1/mn225691/(2)/测试集_复赛/风场2"
num_folders, folders = count_folders(directory_path)

print(f"路径 '{directory_path}' 下共有 {num_folders} 个文件夹。")
print("文件夹列表：", folders)
fn=0
print(directory_path)
for i in folders:
    path=os.path.join(directory_path,i)
    n=0
    print(path)
    for j in os.listdir(path):
        n=n+1
    print(f"路径 '{path}' 下共有 {n} 个文件。")
    fn=fn+n
print("文件夹总数：", fn)  
df5=pd.read_csv('./data/1/mn225691/filtered_1.csv')
print(df5.shape)
print(df5)

import os
import pandas as pd
import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# 读取模板
template_file = "./data/1/mn225691/filtered_1.csv"
df5 = pd.read_csv(template_file)

# 根目录
root = "./data/1/mn225691/(2)/测试集_复赛"

# 遍历每一行
pred_ws, pred_wd = [], []

for idx, row in df5.iterrows():
    farm = row["风场"]
    turbine = row["风机"]
    period = str(row["时段"])
    time_step = int(row["时刻"])

    # 拼接文件路径 (假设 row["风机"] 已经是 m01 这种格式)
    file_path = os.path.join(root, "风场1" if farm=="风场1" else "风场2", turbine, f"{period}.csv")

    if os.path.isfile(file_path):
        try:
            df_csv = pd.read_csv(file_path)

            # 只取需要的列
            required_cols = ["time","变频器电网侧有功功率","外界温度","风速","风向"]
            if not set(required_cols).issubset(df_csv.columns):
                pred_ws.append(np.nan)
                pred_wd.append(np.nan)
                continue

            df_values = df_csv[["风速", "风向"]]

            # 插值补齐
            if df_values.shape[0] != 120 or df_values.isnull().any().any():
                df_values = df_values.ffill().bfill()

            # 转 tensor
            X_input = df_values.values.astype(np.float32)  # [120,2]
            X_tensor = torch.tensor(X_input, dtype=torch.float32).unsqueeze(0).to(device)  # [1,120,2]

            with torch.no_grad():
                pred = model(X_tensor).squeeze(0).cpu().numpy()  # [20,2] → 每 30s 预测1次，10分钟共20个点

            # 找对应时刻的预测值
            # 注意：时刻字段是 30, 60, ..., 600 (秒)
            index = time_step // 30 - 1  # 转换到 0~19
            if 0 <= index < pred.shape[0]:
                ws, wd = pred[index]
                pred_ws.append(float(ws))
                pred_wd.append(float(wd))
            else:
                pred_ws.append(np.nan)
                pred_wd.append(np.nan)

        except Exception as e:
            print(f"❌ 文件处理失败: {file_path}, 错误: {e}")
            pred_ws.append(np.nan)
            pred_wd.append(np.nan)

    else:
        # 文件不存在
        pred_ws.append(np.nan)
        pred_wd.append(np.nan)

# 写回 df5
df5["风速"] = pred_ws
df5["风向"] = pred_wd

# 保存结果
out_file = "./data/1/final_predictions.csv"
df5.to_csv(out_file, index=False, encoding="utf-8-sig")
print("✅ 结果已保存:", out_file)

print(df5.head())

import os
import pandas as pd
import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# 读取模板
template_file = "./data/1/mn225691/filtered_1.csv"
df5 = pd.read_csv(template_file)

# 根目录
root = "./data/1/mn225691/(2)/测试集_复赛"

# 遍历每一行
pred_ws, pred_wd = [], []

# 统计
missing_files, bad_index, bad_cols, bad_shape = 0, 0, 0, 0

for idx, row in df5.iterrows():
    farm = row["风场"]
    turbine = row["风机"]
    period = str(row["时段"])
    time_step = int(row["时刻"])

    # 拼接文件路径
    file_path = os.path.join(root, "风场1" if farm=="风场1" else "风场2", turbine, f"{period}.csv")

    if os.path.isfile(file_path):
        try:
            df_csv = pd.read_csv(file_path)

            # 检查列是否齐全
            required_cols = ["time","变频器电网侧有功功率","外界温度","风速","风向"]
            if not set(required_cols).issubset(df_csv.columns):
                print(f"❌ 列缺失: {file_path}")
                bad_cols += 1
                pred_ws.append(np.nan)
                pred_wd.append(np.nan)
                continue

            df_values = df_csv[["风速", "风向"]]

            # 插值补齐（这里也检查行数）
            if df_values.shape[0] != 120 or df_values.isnull().any().any():
                print(f"⚠️ 行数不等于120或有NaN: {file_path}, shape={df_values.shape}")
                bad_shape += 1
                df_values = df_values.ffill().bfill()

            # 转 tensor
            X_input = df_values.values.astype(np.float32)  # [120,2]
            X_tensor = torch.tensor(X_input, dtype=torch.float32).unsqueeze(0).to(device)  # [1,120,2]

            with torch.no_grad():
                pred = model(X_tensor).squeeze(0).cpu().numpy()  # [20,2]

            # 找对应时刻的预测值
            index = time_step // 30 - 1
            if 0 <= index < pred.shape[0]:
                ws, wd = pred[index]
                pred_ws.append(float(ws))
                pred_wd.append(float(wd))
            else:
                print(f"❌ 时刻越界: {file_path}, 时刻={time_step}, index={index}")
                bad_index += 1
                pred_ws.append(np.nan)
                pred_wd.append(np.nan)

        except Exception as e:
            print(f"❌ 文件处理失败: {file_path}, 错误: {e}")
            pred_ws.append(np.nan)
            pred_wd.append(np.nan)

    else:
        print(f"❌ 文件缺失: {file_path}")
        missing_files += 1
        pred_ws.append(np.nan)
        pred_wd.append(np.nan)

# 写回 df5
df5["风速"] = pred_ws
df5["风向"] = pred_wd

# 保存结果
out_file = "./data/1/final_predictions.csv"
df5.to_csv(out_file, index=False, encoding="utf-8-sig")
print("✅ 结果已保存:", out_file)

# 打印统计
print("\n=== 缺失/损坏统计 ===")
print(f"文件缺失: {missing_files}")
print(f"列缺失: {bad_cols}")
print(f"行数异常或含NaN: {bad_shape}")
print(f"时刻越界: {bad_index}")

print("\n检查完成 ✅")

import pandas as pd

# 读取你的预测结果
# pred_file = "/kaggle/input/mn225691/filtered_1.csv"
# 例如从"m01"变回"m1"
# df5['风机'] = df5['风机'].apply(lambda x: f"m{int(x[1:])}")
# # 例如从"01"变回1（整数类型）
# df5['时段'] = df5['时段'].astype(int)

# # 如果时段之前是字符串格式（如"1"）
# df5['时段'] = df5['时段'].apply(lambda x: str(int(x)))
df_pred = df5
print(df_pred.shape)

# 读取提交模板
template_file = "./data/1/mn225691/filtered_1.csv"
df_template = pd.read_csv(template_file)
print(df_template.shape)

# 1. 检查列数是否一致
if df_pred.shape[1] != df_template.shape[1]:
    print(f"列数不一致！预测文件: {df_pred.shape[1]}, 模板: {df_template.shape[1]}")
else:
    print("列数一致 ✅")

# 2. 检查风场/时段/风机/时刻字段顺序是否一致
fields = ['风场','时段','风机','时刻']
for field in fields:
    if not df_pred[field].equals(df_template[field]):
        print(f"字段 {field} 内容不一致 ❌")
    else:
        print(f"字段 {field} 内容一致 ✅")

# 3. 检查总行数是否一致
if df_pred.shape[0] != df_template.shape[0]:
    print(f"行数不一致！预测文件: {df_pred.shape[0]}, 模板: {df_template.shape[0]}")
else:
    print("总行数一致 ✅")

print(df5.isnull().sum())

nan_count = df5[["风速", "风向"]].isna().sum().sum()
print(df5.shape)
print(f"⚠️ 共有 {nan_count} 个 NaN")

if nan_count > 0:
    print("➡️ 对 NaN 进行插值补全")
    # 按列插值，首尾用前后值填充
    df5["风速"] = df5["风速"].interpolate(limit_direction="both")
    df5["风向"] = df5["风向"].interpolate(limit_direction="both")

    # 再次确认
    nan_count_after = df5[["风速", "风向"]].isna().sum().sum()
    print(f"✅ 插值后 NaN 剩余 {nan_count_after} 个")

# ======================
# 6. 保存结果
# ======================
out_file = "./data/1/final_predictions.csv"
df5.to_csv(out_file, index=False, encoding="utf-8-sig")
print("✅ 结果已保存:", out_file)
print(df5.head())