import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# 重新读取数据并进行详细分析
df = pd.read_csv('data/Accelerometer.csv')

# 1. 时间数据分析与修正
print("=== 1. 时间数据深度分析 ===")

# 分析time列的数值特征（判断是否为微秒级时间戳）
time_values = df['time'].values
print(f"time列数值范围: {time_values.min()} 到 {time_values.max()}")

# 尝试不同的时间戳单位转换
units = ['ns', 'us', 'ms', 's']
for unit in units:
    try:
        # 将时间戳转换为datetime
        converted_time = pd.to_datetime(time_values, unit=unit)
        # 检查时间范围是否合理（2000年以后）
        if converted_time.min() > pd.Timestamp('2000-01-01'):
            print(f"\n成功识别时间戳单位: {unit} (微秒)")
            print(f"转换后的时间范围: {converted_time.min()} 到 {converted_time.max()}")

            # 计算数据采集时长和采样频率
            df['timestamp'] = converted_time
            df = df.sort_values('timestamp').reset_index(drop=True)

            # 计算时间间隔
            time_diff = df['timestamp'].diff().dt.total_seconds()
            valid_time_diff = time_diff[time_diff > 0]  # 过滤掉0和负值

            if len(valid_time_diff) > 0:
                avg_interval = valid_time_diff.mean()
                sampling_rate = 1 / avg_interval
                print(f"数据采集时长: {df['timestamp'].max() - df['timestamp'].min()}")
                print(f"平均采样间隔: {avg_interval:.6f} 秒")
                print(f"实际采样率: {sampling_rate:.1f} Hz")
            break
    except Exception as e:
        continue

# 2. 加速度计数据识别与分析
print(f"\n=== 2. 加速度计数据特征分析 ===")

# 明确x, y, z轴数据列
accel_cols = ['x', 'y', 'z']
print(f"确认加速度计数据列: {accel_cols}")

# 计算各轴的基本统计特征
accel_stats = df[accel_cols].describe()
print(f"\n各轴加速度统计特征:")
print(accel_stats.round(4))

# 分析数据范围，判断是否为原始数据或已校准数据
print(f"\n各轴数据范围分析:")
for col in accel_cols:
    col_min = df[col].min()
    col_max = df[col].max()
    col_range = col_max - col_min
    print(f"{col}轴: {col_min:.4f} ~ {col_max:.4f} (范围: {col_range:.4f})")

# 判断数据类型（原始ADC值或物理单位值）
# 通常手机加速度计原始值范围: ±16384 (14位) 或 ±32768 (16位)
# 物理单位 (m/s²) 范围通常在 ±10 左右（受重力影响）
is_raw_data = any(abs(df[col]).max() > 100 for col in accel_cols)
if is_raw_data:
    print(f"\n数据类型判断: 原始ADC值（数值范围较大）")
    # 估算灵敏度（假设满量程为±8g，1g=9.81m/s²）
    full_scale = 16  # ±8g
    max_raw = max(abs(df[col]).max() for col in accel_cols)
    sensitivity = (full_scale * 9.81) / max_raw
    print(f"估算灵敏度: {sensitivity:.4f} m/s²/LSB")
    # 转换为物理单位
    df[['x_mps2', 'y_mps2', 'z_mps2']] = df[accel_cols] * sensitivity
    accel_phys_cols = ['x_mps2', 'y_mps2', 'z_mps2']
    print(f"转换后物理单位数据范围 (m/s²):")
    for col in accel_phys_cols:
        print(f"  {col}: {df[col].min():.2f} ~ {df[col].max():.2f}")
else:
    print(f"\n数据类型判断: 物理单位值 (m/s²)")
    accel_phys_cols = accel_cols.copy()

# 3. 计算合成加速度（合加速度）
print(f"\n=== 3. 运动特征分析 ===")

# 计算合加速度 (sqrt(x² + y² + z²))
df['accel_mag'] = np.sqrt(df[accel_phys_cols].pow(2).sum(axis=1))

# 分析合加速度特征（静止时应接近9.81 m/s²）
gravity = 9.81
accel_mag_stats = df['accel_mag'].describe()
print(f"合加速度统计 (m/s²):")
print(accel_mag_stats.round(4))
print(f"重力加速度参考值: {gravity} m/s²")

# 判断运动状态
static_threshold = 0.5  # 静止时合加速度波动阈值
accel_mag_std = df['accel_mag'].std()
if accel_mag_std < static_threshold:
    motion_state = "主要为静止状态"
elif accel_mag_std < 2:
    motion_state = "轻微运动状态"
else:
    motion_state = "明显运动状态"

print(f"\n运动状态判断: {motion_state}")
print(f"合加速度标准差: {accel_mag_std:.4f} m/s²")

# 计算静止时的平均重力方向
if motion_state in ["主要为静止状态", "轻微运动状态"]:
    # 取合加速度接近重力值的数据点
    static_data = df[np.abs(df['accel_mag'] - gravity) < static_threshold]
    if len(static_data) > 0:
        avg_gravity_dir = static_data[accel_phys_cols].mean()
        print(f"\n静止时平均重力方向 (m/s²):")
        for col, val in avg_gravity_dir.items():
            print(f"  {col}: {val:.4f}")
        print(f"平均重力方向的合加速度: {np.sqrt(avg_gravity_dir.pow(2).sum()):.4f} m/s²")

# 4. 数据质量分析
print(f"\n=== 4. 数据质量评估 ===")

# 检查数据连续性
time_gaps = df['timestamp'].diff()
large_gaps = time_gaps[time_gaps > pd.Timedelta(seconds=0.1)]  # 超过0.1秒的间隔视为大间隙
print(f"数据总点数: {len(df)}")
print(f"时间间隙超过0.1秒的数量: {len(large_gaps)}")

if len(large_gaps) > 0:
    print(f"最大时间间隙: {large_gaps.max()}")
    print(f"数据完整性: {((len(df) - len(large_gaps)) / len(df) * 100):.2f}%")
else:
    print(f"数据连续性良好，无明显时间间隙")

# 检查异常值（使用3σ准则）
outlier_count = 0
for col in accel_phys_cols:
    mean_val = df[col].mean()
    std_val = df[col].std()
    outliers = df[(df[col] < mean_val - 3 * std_val) | (df[col] > mean_val + 3 * std_val)]
    outlier_count += len(outliers)
    print(f"{col}轴异常值数量: {len(outliers)}")

print(f"总异常值数量: {outlier_count}")
print(f"异常值比例: {(outlier_count / len(df) * 100):.4f}%")