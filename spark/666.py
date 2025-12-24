# 实验七：Spark 实践

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as _sum, avg
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

# 王俊磊 初始环境，加载数据，整合实验
spark = SparkSession.builder \
    .appName("Experiment7_Spark_Practice") \
    .getOrCreate()

data = spark.read.csv(
    "sales_data.csv",
    header=True,
    inferSchema=True
)

print("数据结构：")
data.printSchema()
print("数据预览：")
data.show(5)

# 王启源：使用 DataFrame API 做业务分析
print("各产品类别总销售额分析")
category_revenue = data.groupBy("Product_Category") \
    .agg(_sum("Revenue").alias("Total_Revenue")) \
    .orderBy(col("Total_Revenue").desc())
category_revenue.show()

print("各国家订单总量分析")
country_orders = data.groupBy("Country") \
    .agg(_sum("Order_Quantity").alias("Total_Orders")) \
    .orderBy(col("Total_Orders").desc())
country_orders.show()

print("各产品类别平均单笔订单收入分析")
category_avg_revenue = data.groupBy("Product_Category") \
    .agg(avg("Revenue").alias("Avg_Revenue_Per_Order")) \
    .orderBy(col("Avg_Revenue_Per_Order").desc())
category_avg_revenue.show()

# 任俊毅：Spark SQL 查询
print("使用 Spark SQL 进行查询")
data.createOrReplaceTempView("sales")
# 1. 各国家平均销售额
avg_revenue_sql = spark.sql("""
    SELECT Country, AVG(Revenue) AS Avg_Revenue
    FROM sales
    GROUP BY Country
    ORDER BY Avg_Revenue DESC
""")
avg_revenue_sql.show()
# 2. 销售额最高的国家
top_country_sql = spark.sql("""
    SELECT Country, SUM(Revenue) AS Total_Revenue
    FROM sales
    GROUP BY Country
    ORDER BY Total_Revenue DESC
    LIMIT 1
""")
top_country_sql.show()
# 3. 各国家销售额最高的产品类别
top_category_by_country_sql = spark.sql("""
    SELECT Country, Product_Category, SUM(Revenue) AS Total_Revenue
    FROM sales
    GROUP BY Country, Product_Category
    ORDER BY Country, Total_Revenue DESC
""")
top_category_by_country_sql.show()

# 马浩鑫：MLlib 机器学习
print("使用 Spark MLlib 进行多特征线性回归预测收入revenue")
ml_data = data.select(
    "Order_Quantity",
    "Unit_Price",
    "Unit_Cost",
    "Revenue"
)
# 特征向量
assembler = VectorAssembler(
    inputCols=["Order_Quantity", "Unit_Price", "Unit_Cost"],
    outputCol="features"
)
ml_features = assembler.transform(ml_data) \
    .select("features", "Revenue")
# 划分训练集和测试集
train_data, test_data = ml_features.randomSplit([0.8, 0.2], seed=42)
# 线性回归模型（正则，防止过拟合）
lr = LinearRegression(
    featuresCol="features",
    labelCol="Revenue",
    regParam=0.1
)
lr_model = lr.fit(train_data)
# 预测
predictions = lr_model.transform(test_data)
print("线性回归预测结果（前 5 条）：")
predictions.select("features", "Revenue", "prediction").show(5)
print("模型系数：", lr_model.coefficients)
print("模型截距：", lr_model.intercept)

# 释放资源
spark.stop()