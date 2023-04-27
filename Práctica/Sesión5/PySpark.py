import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import numpy as np

#Creación de la Sesión Spark y lectura csv
spark = SparkSession.builder.appName("Ejemplo PySpark").config("spark.memory.offHeap.enabled","true").config("spark.memory.offHeap.size","10g").getOrCreate()
df = spark.read.option("delimiter", ";").option("header", True).csv("ecommerce.csv")

#Mostramos 5 filas para comprobar que todo está correcto
df.show(5,0)

#Vemos los países que más compras realizan
df.groupBy('Country').agg(countDistinct('CustomerID').alias('country_count')).show()

#Para averiguar cuándo se realizó la última compra en la plataforma, tenemos que convertir la columna "InvoiceDate" a un formato de marca de tiempo y utilizar la función max() de Pyspark:
spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")
df = df.withColumn('date',to_timestamp("InvoiceDate", 'dd/MM/yyyy H:mm'))
df.select(max("date")).show()
#Primera compra
df.select(min("date")).show()


df = df.withColumn("from_date", lit("12/1/10 08:26"))
df = df.withColumn('from_date',to_timestamp("from_date", 'yy/MM/dd HH:mm'))

df2=df.withColumn('from_date',to_timestamp(col('from_date'))).withColumn('recency',col("date").cast("long") - col('from_date').cast("long"))

#Los datos que tenemos no son buenos para segmentar por lo que generaremos frecuencia de compra, gasto y la última vez que compró

df2 = df2.join(df2.groupBy('CustomerID').agg(max('recency').alias('recency')),on='recency',how='leftsemi')

df2.show(5,0)

df2.printSchema()

df_freq = df2.groupBy('CustomerID').agg(count('InvoiceDate').alias('frequency'))

df_freq.show(5,0)

#Unir columnas de consumidor, cantidas y precio
df3 = df2.join(df_freq,on='CustomerID',how='inner')

df3.show(5,0)


#Sacamos valor monetario

m_val = df3.withColumn('TotalAmount',col("Quantity") * col("UnitPrice"))
m_val = m_val.groupBy('CustomerID').agg(sum('TotalAmount').alias('monetary_value'))
finaldf = m_val.join(df3,on='CustomerID',how='inner')

finaldf = finaldf.select(['recency','frequency','monetary_value','CustomerID']).distinct()



#Estandarización
assemble=VectorAssembler(inputCols=[
    'recency','frequency','monetary_value'
], outputCol='features')

assembled_data=assemble.transform(finaldf)

scale=StandardScaler(inputCol='features',outputCol='standardized')
data_scale=scale.fit(assembled_data)
data_scale_output=data_scale.transform(assembled_data)

#Crear modelo de Machine Learning con KMeans

cost = np.zeros(10)

evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='standardized',metricName='silhouette', distanceMeasure='squaredEuclidean')

for i in range(2,10):
    KMeans_algo=KMeans(featuresCol='standardized', k=i)
    KMeans_fit=KMeans_algo.fit(data_scale_output)
    output=KMeans_fit.transform(data_scale_output)
    cost[i] = KMeans_fit.summary.trainingCost