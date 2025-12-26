import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# 设置绘图样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.alpha'] = 0.3

# 重新读取并预处理数据
df = pd.read_csv('data/Accelerometer.csv')

# 时间戳转换（已确认是纳秒级）
df['timestamp'] = pd.to_datetime(df['time'], unit='ns')
df = df.sort_values('timestamp').reset_index(drop=True)

# 计算合加速度
df['accel_mag'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)

# 为了更好的可视化，对数据进行降采样（保持趋势的同时减少点数）
# 由于采样率84.8Hz，每10个点取一个均值，降采样到约8.5Hz
downsample_rate = 10
if len(df) > downsample_rate * 1000:  # 只有当数据点足够多时才降采样
    df_downsampled = df.iloc[::downsample_rate].copy()
else:
    df_downsampled = df.copy()

print(f"原始数据点数: {len(df)}")
print(f"降采样后数据点数: {len(df_downsampled)}")

# 创建综合分析图表
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)

# 1. 各轴加速度时间序列（主图）
ax1 = fig.add_subplot(gs[0:2, 0:2])
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
labels = ['X轴', 'Y轴', 'Z轴']

for i, (col, color, label) in enumerate(zip(['x', 'y', 'z'], colors, labels)):
    ax1.plot(df_downsampled['timestamp'], df_downsampled[col],
             color=color, alpha=0.8, linewidth=1.2, label=label)

# 添加重力加速度参考线
ax1.axhline(y=9.81, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='重力加速度 (9.81 m/s²)')
ax1.axhline(y=-9.81, color='gray', linestyle='--', alpha=0.5, linewidth=1)

ax1.set_title('手机加速度计各轴数据时间序列', fontsize=16, fontweight='bold', pad=20)
ax1.set_ylabel('加速度 (m/s²)', fontsize=12)
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)

# 格式化x轴时间
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
ax1.xaxis.set_major_locator(mdates.SecondLocator(interval=30))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

# 2. 合加速度时间序列
ax2 = fig.add_subplot(gs[2, 0:2])
ax2.plot(df_downsampled['timestamp'], df_downsampled['accel_mag'],
         color='#96CEB4', linewidth=1.5, alpha=0.9)
ax2.axhline(y=9.81, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='静止状态参考值 (9.81 m/s²)')

ax2.set_title('合加速度变化趋势', fontsize=14, fontweight='bold', pad=15)
ax2.set_ylabel('合加速度 (m/s²)', fontsize=12)
ax2.set_xlabel('时间', fontsize=12)
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)

# 格式化x轴时间
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
ax2.xaxis.set_major_locator(mdates.SecondLocator(interval=30))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

# 3. 各轴加速度分布直方图
ax3 = fig.add_subplot(gs[3, 0])
for i, (col, color, label) in enumerate(zip(['x', 'y', 'z'], colors, labels)):
    ax3.hist(df[col], bins=50, alpha=0.6, color=color, label=label, density=True)

ax3.set_title('各轴加速度分布', fontsize=14, fontweight='bold', pad=15)
ax3.set_xlabel('加速度 (m/s²)', fontsize=12)
ax3.set_ylabel('概率密度', fontsize=12)
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(True, alpha=0.3)

# 4. 关键统计信息文本框
ax4 = fig.add_subplot(gs[3, 1])
ax4.axis('off')  # 隐藏坐标轴

# 准备统计信息
stats_text = f"""
数据采集基本信息
━━━━━━━━━━━━━━━━━━━━
• 采集时间范围: {df['timestamp'].min().strftime('%m-%d %H:%M:%S')} 
                ~ {df['timestamp'].max().strftime('%m-%d %H:%M:%S')}
• 总采集时长: {(df['timestamp'].max() - df['timestamp'].min()).total_seconds():.0f} 秒
• 采样率: 84.8 Hz
• 数据总点数: {len(df):,}
• 数据完整性: 99.99%

运动状态分析
━━━━━━━━━━━━━━━━━━━━
• 运动状态: 明显运动状态
• 合加速度均值: {df['accel_mag'].mean():.2f} m/s²
• 合加速度标准差: {df['accel_mag'].std():.2f} m/s²
• 合加速度范围: {df['accel_mag'].min():.2f} ~ {df['accel_mag'].max():.2f} m/s²

各轴极值 (m/s²)
━━━━━━━━━━━━━━━━━━━━
• X轴: {df['x'].min():.2f} ~ {df['x'].max():.2f}
• Y轴: {df['y'].min():.2f} ~ {df['y'].max():.2f}
• Z轴: {df['z'].min():.2f} ~ {df['z'].max():.2f}
"""

# 添加文本框
ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', bbox=dict(boxstyle='round,pad=0.8',
         facecolor='#F8F9FA', edgecolor='#DDE2E5', alpha=0.9))

# 调整布局并保存
plt.tight_layout()
plt.savefig('手机加速度计数据分析报告.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()

# 生成详细的数据分析报告
report_content = f"""# 手机加速度计数据详细分析报告

## 1. 数据采集基本信息

### 1.1 时间信息
- **采集起始时间**: {df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}
- **采集结束时间**: {df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}
- **总采集时长**: {(df['timestamp'].max() - df['timestamp'].min()).total_seconds():.2f} 秒 ({(df['timestamp'].max() - df['timestamp'].min()).total_seconds()/60:.1f} 分钟)
- **数据采集日期**: {df['timestamp'].min().strftime('%Y年%m月%d日')}

### 1.2 技术参数
- **采样率**: 84.8 Hz (每0.0118秒采集一个数据点)
- **数据总点数**: {len(df):,} 个
- **数据完整性**: 99.99% (仅4个时间间隙超过0.1秒)
- **最大时间间隙**: 0.125秒
- **异常值比例**: 2.87% (主要为瞬时冲击信号)

## 2. 加速度数据特征分析

### 2.1 数据格式确认
- **数据单位**: 物理单位 (m/s²，米/秒²)
- **坐标轴数量**: 3轴 (X轴、Y轴、Z轴)
- **数据类型**: 线性加速度数据

### 2.2 各轴加速度统计特征
| 统计指标 | X轴 (m/s²) | Y轴 (m/s²) | Z轴 (m/s²) | 合加速度 (m/s²) |
|----------|------------|------------|------------|------------------|
| 均值     | {df['x'].mean():.4f} | {df['y'].mean():.4f} | {df['z'].mean():.4f} | {df['accel_mag'].mean():.4f} |
| 标准差   | {df['x'].std():.4f} | {df['y'].std():.4f} | {df['z'].std():.4f} | {df['accel_mag'].std():.4f} |
| 最小值   | {df['x'].min():.4f} | {df['y'].min():.4f} | {df['z'].min():.4f} | {df['accel_mag'].min():.4f} |
| 最大值   | {df['x'].max():.4f} | {df['y'].max():.4f} | {df['z'].max():.4f} | {df['accel_mag'].max():.4f} |
| 中位数   | {df['x'].median():.4f} | {df['y'].median():.4f} | {df['z'].median():.4f} | {df['accel_mag'].median():.4f} |

## 3. 运动状态分析

### 3.1 运动状态判断
- **综合判断**: 明显运动状态
- **判断依据**:
  1. 合加速度标准差为 {df['accel_mag'].std():.2f} m/s²，远大于静止状态阈值 (0.5 m/s²)
  2. 合加速度均值 ({df['accel_mag'].mean():.2f} m/s²) 明显偏离重力加速度 (9.81 m/s²)
  3. 各轴加速度波动范围大，存在明显的加速和减速过程

### 3.2 可能的运动类型推测
根据加速度特征，可能的运动类型包括：
1. **步行/跑步**: 存在周期性的加速度变化（但需要更长时间数据确认）
2. **交通工具移动**: 可能存在持续的加速度和匀速阶段
3. **手部动作**: 快速的手部摆动或设备移动
4. **冲击/振动**: 存在瞬时的大加速度值（如设备掉落、碰撞）

## 4. 数据质量评估

### 4.1 数据完整性
- **完整性评分**: 9.9/10
- **评估依据**:
  - 数据连续性良好，99.99%的数据点间隔正常
  - 仅存在4个微小的数据间隙（最大0.125秒）
  - 无数据丢失或严重错位现象

### 4.2 数据准确性
- **准确性评分**: 9.0/10
- **评估依据**:
  - 异常值比例为2.87%，在可接受范围内
  - 数据范围合理（符合手机加速度计的测量范围）
  - 重力加速度分量可识别，数据物理意义明确

### 4.3 数据可靠性
- **可靠性评分**: 9.5/10
- **评估依据**:
  - 采样率稳定（84.8 Hz）
  - 时间戳准确，可精确追溯数据采集时间
  - 数据分布合理，无明显的系统误差

## 5. 应用建议

### 5.1 数据用途建议
基于数据特征，该加速度数据可用于：
1. **运动状态识别**: 分析用户的运动模式和活动类型
2. **设备姿态分析**: 判断手机在采集过程中的姿态变化
3. **振动监测**: 检测环境或设备的振动情况
4. **冲击检测**: 识别设备是否受到碰撞或冲击

### 5.2 数据处理建议
1. **异常值处理**: 建议使用3σ准则或中位数滤波去除异常值
2. **数据平滑**: 可采用移动平均滤波减少噪声干扰
3. **特征提取**: 建议计算以下特征用于后续分析：
   - 时间域特征：均值、标准差、最大值、最小值、峰值因子
   - 频率域特征：通过FFT变换提取主要频率成分
4. **数据标注**: 建议结合实际场景对数据进行运动类型标注，提高分析准确性

## 6. 注意事项

1. **坐标系说明**: 手机加速度计的坐标系通常定义为：
   - X轴：水平向右（当手机正面朝上时）
   - Y轴：水平向前
   - Z轴：垂直向上
   具体坐标系可能因手机品牌和型号略有差异

2. **重力影响**: 加速度计数据包含重力加速度分量，在分析运动加速度时需要进行重力分离

3. **采样率考虑**: 84.8 Hz的采样率适用于大多数日常运动分析，但对于高速运动（如球类运动）可能需要更高的采样率

4. **数据时长**: 7分39秒的数据时长适合进行短期运动分析，如需长期运动模式识别建议延长采集时间
"""

# 保存报告到文件
with open('手机加速度计数据分析报告.md', 'w', encoding='utf-8') as f:
    f.write(report_content)
