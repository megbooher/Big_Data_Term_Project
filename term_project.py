import findspark
findspark.init() 
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext

sc = SparkContext()
sqlContext = SQLContext(sc)

company_df = sqlContext.read.format('com.databricks.spark.csv').options(header='true', inferschema='true').load('./fortune500.csv')

company_df.cache()
company_df.printSchema()

company_df.describe().toPandas().transpose()

import pandas as pdnumeric_features 
pdnumeric_features = [t[0] for t in company_df.dtypes if t[1] == 'int' or t[1] == 'double']
sampled_data = company_df.select(numeric_features).sample(False, 0.8).toPandas()
axs = pd.scatter_matrix(sampled_data, figsize=(10, 10))
n = len(sampled_data.columns)
for i in range(n):
    v = axs[i, 0]
    v.yaxis.label.set_rotation(0)
    v.yaxis.label.set_ha('right')
    v.set_yticks(())
    h = axs[n-1, i]
    h.xaxis.label.set_rotation(90)
    h.set_xticks(())