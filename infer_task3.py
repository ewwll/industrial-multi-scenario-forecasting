import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
# from torch.utils.tensorboard import SummaryWriter

# ======================
# 1. 数据预处理
# ======================
df = pd.read_csv('./data/3/3167182637813/.csv')

# 唯一主键
df['主键'] = df['工厂编码'].astype(str) + "_" + df['物料编码'].astype(str)

# 日期转年月
df['过账日期'] = pd.to_datetime(df['过账日期'])
df['年月'] = df['过账日期'].dt.to_period('M').dt.to_timestamp()

# 月级聚合
df_month = (
    df.groupby(['主键','工厂编码','物料编码','物料品牌','物料类型','物料品类','年月'])['需求量']
    .sum()
    .reset_index()
)

# 透视成宽表
df_month_pivot = df_month.pivot_table(
    index=['主键','工厂编码','物料编码','物料品牌','物料类型','物料品类'],
    columns='年月',
    values='需求量'
).reset_index()

df_month_pivot.columns = [str(c) for c in df_month_pivot.columns]

# 缺失值补0
df_month_pivot_filled = df_month_pivot.fillna(0.0)

# 时间列
static_cols = ['主键','工厂编码','物料编码','物料品牌','物料类型','物料品类']
time_cols = [c for c in df_month_pivot_filled.columns if c not in static_cols]

# 转换为矩阵 (N, T)
series_matrix = df_month_pivot_filled[time_cols].values

# 标准化
scaler = StandardScaler()
series_matrix_scaled = scaler.fit_transform(series_matrix.reshape(-1,1)).reshape(series_matrix.shape)

print("series_matrix_scaled:", series_matrix_scaled.shape)
df10=pd.read_csv('./data/3/mn225691/_.csv')
print(df10.head())

# ======================
# 2. 数据集定义
# ======================
class MultiVarGlobalDataset(Dataset):
    def __init__(self, data, seq_len=12, pred_len=3):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.T = data.shape[1]

        self.indices = [
            i for i in range(self.T - seq_len - pred_len + 1)
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.data[:, i:i+self.seq_len]   # (N, seq_len)
        y = self.data[:, i+self.seq_len:i+self.seq_len+self.pred_len]  # (N, pred_len)
        return torch.tensor(x.T, dtype=torch.float32), torch.tensor(y.T, dtype=torch.float32)


# ======================
# 3. DLinear 模型
# ======================
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os

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


# ======================
# 4. 加权 Loss
# ======================
class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = torch.tensor(weights).float()

    def forward(self, yhat, y):
        loss = 0
        for i, w in enumerate(self.weights):
            loss += w * torch.mean((yhat[:, i, :] - y[:, i, :]) ** 2)
        return loss


# ======================
# 5. 训练
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 12
PRED_LEN = 3
BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-3
N_SPLITS = 5

class Configs:
    def __init__(self, seq_len, pred_len, enc_in, individual=False):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.individual = individual

dataset = MultiVarGlobalDataset(series_matrix_scaled, seq_len=SEQ_LEN, pred_len=PRED_LEN)
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

configs = Configs(seq_len=SEQ_LEN, pred_len=PRED_LEN, enc_in=series_matrix_scaled.shape[0], individual=False)

best_model = None
best_loss = float("inf")

for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(dataset)))):
    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    model = Model(configs).to(DEVICE)
    #model = DLinearModel(seq_len=SEQ_LEN, pred_len=PRED_LEN, enc_in=series_matrix_scaled.shape[0]).to(DEVICE)
    criterion = WeightedMSELoss(weights=[0.2,0.2,0.6])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS+1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            yhat = model(xb)
            loss = criterion(yhat, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        mean_train_loss = np.mean(train_losses)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                yhat = model(xb)
                loss = criterion(yhat, yb)
                val_losses.append(loss.item())
        mean_val_loss = np.mean(val_losses)

        print(f"[Fold {fold}] Epoch {epoch:02d} | Train Loss: {mean_train_loss:.6f} | Val Loss: {mean_val_loss:.6f}")

        if mean_val_loss < best_loss:
            best_loss = mean_val_loss
            best_model = model

print(f"✅ 最佳模型 Val Loss: {best_loss:.6f}")


# ======================
# 6. 最终预测 (只取 M+3)
# ======================
best_model.eval()
x_in = torch.tensor(series_matrix_scaled[:, -SEQ_LEN:], dtype=torch.float32).T.unsqueeze(0).to(DEVICE)  # (1, seq_len, N)

with torch.no_grad():
    yhat = best_model(x_in)  # (1, 3, N)
    yhat = yhat.cpu().numpy().squeeze(0)  # (3, N)

# 逆标准化
last_seq = series_matrix_scaled[:, -SEQ_LEN:]
full_seq = np.concatenate([last_seq, yhat.T], axis=1)
full_seq_inv = scaler.inverse_transform(full_seq.T).T
yhat_inv = full_seq_inv[:, -3:]  # (N, 3)

# 只取 M+3
yhat_m3 = yhat_inv[:, -1]

# ======================
# 7. 生成提交文件
# ======================
submit = df_month_pivot_filled[['工厂编码','物料编码']].copy()
submit['M+3月预测需求量'] = yhat_m3
submit['物料编码'] = submit['物料编码'].astype(int)

submit.to_csv("submission_demand.csv", index=False, encoding="utf-8-sig")
print(submit.head())

are_equal = (submit['工厂编码'].equals(df10['工厂编码']) and submit['物料编码'].equals(df10['物料编码']))

if are_equal:
    print("两个 DataFrame 的 '工厂编码' 和 '物料编码' 列内容完全相同，且顺序一致。")
else:
    print("两个 DataFrame 的 '工厂编码' 和 '物料编码' 列内容不相同或顺序不一致。")

print(df10.shape)
print(submit.shape)
torch.save(model.state_dict(), "last_dlinear_demand.pth")
print("✅ 最终模型已保存为 last_dlinear_demand.pth")

# 假设 submit 和 df10 维度相同
# ------------------------------------------------------------------

# 1. 创建两个布尔 Series，检查 '工厂编码' 和 '物料编码' 是否相等
factory_mismatch = (submit['工厂编码'] != df10['工厂编码'])
material_mismatch = (submit['物料编码'] != df10['物料编码'])

# 2. 找出任一列不匹配的行（使用逻辑 OR |）
mismatched_rows = factory_mismatch | material_mismatch

# 3. 筛选出不匹配的行，并只显示关键列
# 在 submit 中查看那些不匹配的行
print("--- 在 submit 中，不匹配的行详情：---")
print(submit[mismatched_rows])

# 在 df10 中查看那些不匹配的行 (作为对比)
print("\n--- 在 df10 中，对应的不匹配行详情：---")
print(df10[mismatched_rows])

# 4. 统计不匹配的行数
print(f"\n总共有 {mismatched_rows.sum()} 行的 '工厂编码' 或 '物料编码' 不匹配。")

# 1. 创建用于比较的新 DataFrame
# 确保只包含需要比较的列
submit_keys = submit[['工厂编码', '物料编码']].copy()
df10_keys = df10[['工厂编码', '物料编码']].copy()

# 2. 核心步骤：对两者按相同的复合键进行排序，并重置索引
submit_sorted = submit_keys.sort_values(by=['工厂编码', '物料编码']).reset_index(drop=True)
df10_sorted = df10_keys.sort_values(by=['工厂编码', '物料编码']).reset_index(drop=True)

# 3. 重新执行 equals 检查
# .equals() 方法将检查 内容、数据类型 和 索引 是否完全一致
are_content_equal = submit_sorted.equals(df10_sorted)

print(f"原始维度: submit({submit.shape}), df10({df10.shape})")

if are_content_equal:
    print("✅ 两个 DataFrame 包含完全相同的键值对集合（内容完全相同，已忽略顺序）。")
else:
    print("❌ 即使忽略顺序，两个 DataFrame 的键值对集合仍存在差异。")
    
    # 进一步诊断差异（如果为 False）
    # 找到 submit 中有，但 df10 中没有的行
    merged_df = pd.merge(submit_keys, df10_keys, on=['工厂编码', '物料编码'], how='outer', indicator=True)
    
    # 找出仅存在于 submit 或 仅存在于 df10 的键
    unique_to_submit = merged_df[merged_df['_merge'] == 'left_only']
    unique_to_df10 = merged_df[merged_df['_merge'] == 'right_only']
    
    print(f"\n差异诊断:")
    print(f"  - 仅存在于 submit 中的键数量: {len(unique_to_submit)}")
    print(f"  - 仅存在于 df10 中的键数量: {len(unique_to_df10)}")
    
    if not unique_to_submit.empty:
        print("\n仅存在于 submit 中的键 (前5个):")
        print(unique_to_submit[['工厂编码', '物料编码']].head())

print("Submit Dtypes:")
print(submit[['工厂编码', '物料编码']].dtypes)
print("\nDf10 Dtypes:")
print(df10[['工厂编码', '物料编码']].dtypes)

# 假设 submit 和 df10 已经存在

# 1. 创建用于比较的新 DataFrame，并强制将 '物料编码' 统一为 int64 或 str
# 推荐统一为 int64，因为它们本身是整数
submit_keys = submit[['工厂编码', '物料编码']].copy()
df10_keys = df10[['工厂编码', '物料编码']].copy()

# ⚠️ 关键修正：统一物料编码的 Dtype
submit_keys['物料编码'] = submit_keys['物料编码'].astype('int64')

# 2. 核心步骤：对两者按相同的复合键进行排序，并重置索引
submit_sorted = submit_keys.sort_values(by=['工厂编码', '物料编码']).reset_index(drop=True)
df10_sorted = df10_keys.sort_values(by=['工厂编码', '物料编码']).reset_index(drop=True)

# 3. 重新执行 equals 检查
are_content_equal = submit_sorted.equals(df10_sorted)

if are_content_equal:
    print("✅ 强制统一 Dtype 后，键集合完全匹配！")
else:
    # 理论上到这一步不会再出现 False
    print("❌ Dtype 统一后仍不匹配，存在其他未知差异。")