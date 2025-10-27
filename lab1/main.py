import pandas as pd
import numpy as np

data = pd.read_csv("./data.csv",encoding="gbk")
data_1 = data.dropna(how="any")
print(data_1)

data_before_filter = data_1
data_after_filter1 = data_1.loc[data_before_filter["traffic"] != 0]
data_after_filter2 = data_after_filter1.loc[data_after_filter1["from_level"] == '一般节点']

print(data_after_filter2)

data_before_sample = data_after_filter2
columns = data_before_sample.columns
weight_sample = data_before_sample.copy()
weight_sample['weight'] = 0

for i in weight_sample.index:
    if weight_sample.at[i, 'to_level'] == '一般节点':
        weight = 1
    else:
        weight = 5
    weight_sample.at[i, 'weight'] = weight

weight_sample_finish = weight_sample.sample(n=50, weights='weight')
print(weight_sample_finish)
weight_sample_finish = weight_sample[columns]
print(weight_sample_finish)

random_sample = data_before_sample
random_sample_finish = random_sample.sample(n=50)
print(random_sample_finish)
random_sample_finish = random_sample_finish[columns]
print(random_sample_finish)

ybjd = data_before_sample.loc[data_before_sample['to_level'] == '一般节点']
wlhx = data_before_sample.loc[data_before_sample['to_level'] == '网络核心']

after_sample = pd.concat([ybjd.sample(17), wlhx.sample(33)])
print(after_sample)