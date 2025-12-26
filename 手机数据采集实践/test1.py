import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

# 读取CSV文件
df = pd.read_csv('data/Accelerometer.csv')

# 查看数据基本信息
print("=== 数据基本信息 ===")
print(f"数据形状: {df.shape} (行数, 列数)")
print(f"\n列名: {list(df.columns)}")
print(f"\n数据类型:")
print(df.dtypes)
print(f"\n前5行数据:")
print(df.head())

# 检查缺失值
print(f"\n=== 缺失值统计 ===")
missing_data = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    '缺失数量': missing_data,
    '缺失百分比(%)': missing_percent.round(2)
})
print(missing_df[missing_df['缺失数量'] > 0])

# 查看数据统计描述
print(f"\n=== 数据统计描述 ===")
print(df.describe())

# 检查时间列（如果存在）
time_columns = [col for col in df.columns if 'time' in col.lower() or 'timestamp' in col.lower()]
print(f"\n=== 时间相关列 ===")
if time_columns:
    print(f"找到时间相关列: {time_columns}")
    for col in time_columns[:1]:  # 只查看第一个时间列
        print(f"\n{col}列的前5个值:")
        print(df[col].head())
        # 尝试转换为datetime格式
        try:
            # 检查是否为时间戳（数字格式）
            if pd.api.types.is_numeric_dtype(df[col]):
                print(f"{col}列是数字格式，尝试转换为datetime（假设是毫秒级时间戳）")
                df[f'{col}_converted'] = pd.to_datetime(df[col], unit='ms')
                print(f"转换后的前5个值:")
                print(df[f'{col}_converted'].head())
                print(f"时间范围: {df[f'{col}_converted'].min()} 到 {df[f'{col}_converted'].max()}")
                print(f"数据采集时长: {df[f'{col}_converted'].max() - df[f'{col}_converted'].min()}")
            else:
                df[f'{col}_converted'] = pd.to_datetime(df[col])
                print(f"转换后的前5个值:")
                print(df[f'{col}_converted'].head())
        except Exception as e:
            print(f"转换时间格式失败: {e}")
else:
    print("未找到明显的时间相关列")

# 识别加速度计数据列（通常包含x, y, z）
accel_columns = []
for col in df.columns:
    col_lower = col.lower()
    if any(keyword in col_lower for keyword in ['x', 'y', 'z']) and \
       any(accel_keyword in col_lower for accel_keyword in ['accel', 'acceleration', 'gyro', 'gyroscope', 'sensor']):
        accel_columns.append(col)

if not accel_columns:
    # 如果没有明确标识，查找数值列中可能的加速度数据
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # 加速度数据通常在一定范围内（如-10到10 m/s²或-16384到16384原始值）
    for col in numeric_cols:
        col_range = df[col].max() - df[col].min()
        if 1 < col_range < 20 or 1000 < col_range < 32768:  # 粗略判断
            accel_columns.append(col)

print(f"\n=== 识别的加速度计相关列 ===")
if accel_columns:
    print(f"可能的加速度/陀螺仪数据列: {accel_columns}")
    # 显示这些列的统计信息
    print(f"\n加速度计数据统计:")
    print(df[accel_columns].describe())
else:
    print("未识别出明显的加速度计数据列")
    accel_columns = df.select_dtypes(include=[np.number]).columns.tolist()[:3]  # 取前3个数值列作为备选
    print(f"备选数值列（前3个）: {accel_columns}")

# 查看数据采集频率（如果有时间列）
if 'time_converted' in df.columns or any('_converted' in col for col in df.columns):
    time_col = [col for col in df.columns if '_converted' in col][0]
    # 计算时间间隔
    df = df.sort_values(time_col).reset_index(drop=True)
    time_diff = df[time_col].diff().dt.total_seconds()
    sampling_interval = time_diff.mean()
    sampling_rate = 1 / sampling_interval if sampling_interval > 0 else 0
    print(f"\n=== 数据采集频率 ===")
    print(f"平均采样间隔: {sampling_interval:.6f} 秒")
    print(f"估计采样率: {sampling_rate:.1f} Hz")