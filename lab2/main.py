import pandas as pd
import matplotlib.pyplot as plt
from pandas import to_numeric

data = pd.read_csv("./Pokemon.csv",encoding="Windows-1252")

##删除多余的行和列
data.drop(columns="#", inplace=True)
data.drop(index=[808, 809], inplace=True)

#去除type2列的异常值
data['Type 2'].value_counts().plot(kind='bar')
drop_types = ['Bug', 'A', '273', '0', 'BBB']
for type in drop_types:
    index = data[data["Type 2"] == type].index
    data.drop(index=index, inplace=True)

#去除重复的行
data.drop_duplicates(inplace=True)

#将Generation列和Legendary列进行一个向前填充
data["Generation"] = data["Generation"].fillna(method="ffill")
data["Legendary"] = data["Legendary"].fillna(method="ffill")

nums = ["1", "2","3","4","5","6","7","8","9"]

data = data.reset_index(drop=True)

#将generation列和Legendary列的异常值修改
for index, value in data["Generation"].items():
    if value not in nums:
        #print(index, value)
        data.at[index, "Generation"] = data.at[index-1, "Generation"]

bools = ["TRUE", "FALSE"]
for index, value in data["Legendary"].items():
    if value not in bools:
        #print(index, value)
        data.at[index, "Legendary"] = data.at[index-1, "Legendary"]

#将列的数据类型修正
to_numeric_columns = ['Total', 'HP', 'Attack', 'Sp. Atk', 'Sp. Def', 'Speed','Defense', 'Generation']
data.drop(index=794,inplace=True)
for col in to_numeric_columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

#填充缺失值
data.dropna(subset=["Name"], inplace=True)
data["HP"] = data["HP"].fillna(data['HP'].mean())
data["Type 2"] = data["Type 2"].fillna(method="ffill")


#plt.scatter(range(0, data.shape[0]), data.iloc[:, 6])
#plt.show()
