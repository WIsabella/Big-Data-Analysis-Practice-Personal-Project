import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (16, 10)

# 1. 读取并分析Location.csv文件
print("=== 1. 读取位置数据 (Location.csv) ===")
loc_df = pd.read_csv('data/Location.csv')

# 查看数据基本信息
print(f"数据形状: {loc_df.shape} (行数, 列数)")
print(f"\n列名: {list(loc_df.columns)}")
print(f"\n前5行数据:")
print(loc_df.head())

# 查看数据类型和缺失值
print(f"\n数据类型:")
print(loc_df.dtypes)

print(f"\n缺失值统计:")
missing_data = loc_df.isnull().sum()
missing_percent = (loc_df.isnull().sum() / len(loc_df)) * 100
missing_df = pd.DataFrame({
    '缺失数量': missing_data,
    '缺失百分比(%)': missing_percent.round(2)
})
print(missing_df[missing_df['缺失数量'] > 0])

# 2. 识别关键位置信息列
print(f"\n=== 2. 识别位置数据关键列 ===")
# 常见的位置数据列名
location_cols = {
    'latitude': None,  # 纬度
    'longitude': None,  # 经度
    'altitude': None,  # 海拔
    'time': None,  # 时间戳
    'speed': None,  # 速度
    'accuracy': None  # 定位精度
}

# 自动匹配列名
for col in loc_df.columns:
    col_lower = col.lower()
    if any(keyword in col_lower for keyword in ['lat', '纬度']):
        location_cols['latitude'] = col
    elif any(keyword in col_lower for keyword in ['lon', 'lng', '经度']):
        location_cols['longitude'] = col
    elif any(keyword in col_lower for keyword in ['alt', '海拔']):
        location_cols['altitude'] = col
    elif any(keyword in col_lower for keyword in ['time', '时间', 'timestamp']):
        location_cols['time'] = col
    elif any(keyword in col_lower for keyword in ['speed', '速度']):
        location_cols['speed'] = col
    elif any(keyword in col_lower for keyword in ['acc', '精度', 'accuracy']):
        location_cols['accuracy'] = col

print("识别到的位置数据列:")
for key, value in location_cols.items():
    if value:
        print(f"  {key}: {value}")
    else:
        print(f"  {key}: 未找到")

# 3. 时间戳处理（与之前的加速度计数据时间对齐）
print(f"\n=== 3. 时间戳处理与数据对齐 ===")
# 先读取之前的加速度计数据
accel_df = pd.read_csv('data/Accelerometer.csv')
# 转换加速度计数据的时间戳（已确认是纳秒级）
accel_df['timestamp'] = pd.to_datetime(accel_df['time'], unit='ns')
print(f"加速度计数据时间范围:")
print(f"  起始: {accel_df['timestamp'].min()}")
print(f"  结束: {accel_df['timestamp'].max()}")
print(f"  时长: {(accel_df['timestamp'].max() - accel_df['timestamp'].min()).total_seconds():.0f}秒")

# 处理位置数据的时间戳
if location_cols['time']:
    time_col = location_cols['time']
    print(f"\n位置数据时间列: {time_col}")

    # 检查时间列的数据类型
    if pd.api.types.is_numeric_dtype(loc_df[time_col]):
        # 尝试不同的时间戳单位
        units = ['ns', 'us', 'ms', 's']
        for unit in units:
            try:
                loc_df['timestamp'] = pd.to_datetime(loc_df[time_col], unit=unit)
                if loc_df['timestamp'].min() > pd.Timestamp('2020-01-01'):
                    print(f"  时间戳单位识别: {unit}")
                    break
            except:
                continue
    else:
        # 尝试直接转换为datetime
        try:
            loc_df['timestamp'] = pd.to_datetime(loc_df[time_col])
            print(f"  时间格式识别: 直接转换")
        except:
            print(f"  时间格式识别失败，使用默认索引时间")
            # 创建虚拟时间（假设均匀采样）
            start_time = accel_df['timestamp'].min()
            end_time = accel_df['timestamp'].max()
            loc_df['timestamp'] = pd.date_range(start=start_time, end=end_time, periods=len(loc_df))

    print(f"位置数据时间范围:")
    print(f"  起始: {loc_df['timestamp'].min()}")
    print(f"  结束: {loc_df['timestamp'].max()}")
    print(f"  时长: {(loc_df['timestamp'].max() - loc_df['timestamp'].min()).total_seconds():.0f}秒")

    # 检查两个数据集的时间重叠情况
    time_overlap_start = max(accel_df['timestamp'].min(), loc_df['timestamp'].min())
    time_overlap_end = min(accel_df['timestamp'].max(), loc_df['timestamp'].max())
    overlap_duration = (time_overlap_end - time_overlap_start).total_seconds()

    print(f"\n时间重叠情况:")
    print(f"  重叠起始: {time_overlap_start}")
    print(f"  重叠结束: {time_overlap_end}")
    print(f"  重叠时长: {overlap_duration:.0f}秒")

    if overlap_duration < 10:
        print("警告: 两个数据集的时间重叠较少，可能影响联合分析效果")
else:
    print("未找到时间列，无法进行时间对齐")
    # 创建虚拟时间
    start_time = accel_df['timestamp'].min()
    end_time = accel_df['timestamp'].max()
    loc_df['timestamp'] = pd.date_range(start=start_time, end=end_time, periods=len(loc_df))

# 4. 位置数据核心特征分析
print(f"\n=== 4. 位置数据核心特征分析 ===")
# 提取关键位置数据
if location_cols['latitude'] and location_cols['longitude']:
    lat_col = location_cols['latitude']
    lon_col = location_cols['longitude']


    # 计算跑步距离（使用Haversine公式）
    def haversine_distance(lat1, lon1, lat2, lon2):
        """计算两点之间的球面距离（米）"""
        R = 6371000  # 地球半径（米）
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c


    # 计算累计距离
    loc_df_sorted = loc_df.sort_values('timestamp').reset_index(drop=True)
    distances = [0]  # 初始距离为0

    for i in range(1, len(loc_df_sorted)):
        lat1 = loc_df_sorted[lat_col].iloc[i - 1]
        lon1 = loc_df_sorted[lon_col].iloc[i - 1]
        lat2 = loc_df_sorted[lat_col].iloc[i]
        lon2 = loc_df_sorted[lon_col].iloc[i]

        # 过滤无效的经纬度数据
        if (not np.isnan(lat1)) and (not np.isnan(lon1)) and (not np.isnan(lat2)) and (not np.isnan(lon2)):
            dist = haversine_distance(lat1, lon1, lat2, lon2)
            # 过滤异常大的距离（可能是定位错误）
            if dist < 100:  # 单次移动不超过100米（跑步场景合理范围）
                distances.append(distances[-1] + dist)
            else:
                distances.append(distances[-1])
        else:
            distances.append(distances[-1])

    loc_df_sorted['cumulative_distance'] = distances
    total_distance = loc_df_sorted['cumulative_distance'].max()

    print(f"跑步距离分析:")
    print(f"  总跑步距离: {total_distance:.2f}米 ({total_distance / 1000:.2f}公里)")
    print(f"  位置数据点数: {len(loc_df_sorted)}")

    # 计算平均速度
    if location_cols['speed']:
        speed_col = location_cols['speed']
        # 过滤无效速度数据（假设跑步速度在0-15 m/s之间，即0-54 km/h）
        valid_speed = loc_df_sorted[speed_col][(loc_df_sorted[speed_col] >= 0) & (loc_df_sorted[speed_col] <= 15)]
        if len(valid_speed) > 0:
            avg_speed = valid_speed.mean()
            print(f"速度分析 (基于位置数据):")
            print(f"  平均速度: {avg_speed:.2f} m/s ({avg_speed * 3.6:.1f} km/h)")
            print(f"  最大速度: {valid_speed.max():.2f} m/s ({valid_speed.max() * 3.6:.1f} km/h)")
            print(f"  最小速度: {valid_speed.min():.2f} m/s ({valid_speed.min() * 3.6:.1f} km/h)")

    # 海拔分析
    if location_cols['altitude']:
        alt_col = location_cols['altitude']
        valid_alt = loc_df_sorted[alt_col][~np.isnan(loc_df_sorted[alt_col])]
        if len(valid_alt) > 0:
            alt_change = valid_alt.max() - valid_alt.min()
            print(f"海拔分析:")
            print(f"  平均海拔: {valid_alt.mean():.2f}米")
            print(f"  海拔变化范围: {valid_alt.min():.2f} ~ {valid_alt.max():.2f}米 (变化{alt_change:.2f}米)")

# 5. 加速度计数据重新预处理（用于联合分析）
print(f"\n=== 5. 加速度计数据重新预处理 ===")
# 计算合加速度
accel_df['accel_mag'] = np.sqrt(accel_df['x'] ** 2 + accel_df['y'] ** 2 + accel_df['z'] ** 2)

# 降采样加速度计数据（与位置数据频率匹配）
# 计算位置数据的采样间隔
loc_time_diff = loc_df_sorted['timestamp'].diff().dt.total_seconds()
loc_sampling_interval = loc_time_diff[loc_time_diff > 0].mean() if len(loc_time_diff[loc_time_diff > 0]) > 0 else 1.0

# 加速度计数据降采样
accel_sampling_interval = 1 / 84.8  # 之前计算的84.8 Hz
downsample_factor = int(round(loc_sampling_interval / accel_sampling_interval))
if downsample_factor < 1:
    downsample_factor = 1

accel_df_sorted = accel_df.sort_values('timestamp').reset_index(drop=True)
# 每N个点取一个均值
accel_df_downsampled = accel_df_sorted.iloc[::downsample_factor].copy()

print(f"加速度计数据降采样:")
print(f"  原始数据点数: {len(accel_df_sorted)}")
print(f"  降采样后点数: {len(accel_df_downsampled)}")
print(f"  降采样因子: {downsample_factor}")

# 保存处理后的数据供后续可视化使用
loc_df_processed = loc_df_sorted.copy()
accel_df_processed = accel_df_downsampled.copy()

print(f"\n=== 6. 联合分析准备完成 ===")
print(f"位置数据: {len(loc_df_processed)}个位置点，总距离{total_distance:.2f}米")
print(
    f"加速度数据: {len(accel_df_processed)}个采样点，覆盖时间{(accel_df_processed['timestamp'].max() - accel_df_processed['timestamp'].min()).total_seconds():.0f}秒")
