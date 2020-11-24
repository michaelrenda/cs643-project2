from pyspark.sql import SparkSession
import pyspark.sql as sql
from pyspark.sql.types import DoubleType
import os
from pyspark.ml.feature import RFormula
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression

# Creates a session on a local master
session = SparkSession.builder.appName("CSV to Dataset").master("local[*]").getOrCreate()

# Reads a CSV file with header, called books.csv, stores it in a dataframe
df = session.read.csv(header=True, inferSchema=True, path='TrainingDataset.csv')
df2 = df.withColumn("label", df["quality"].cast(DoubleType()))
# Shows at most 5 rows from the dataframe
df2.show(5)
assembler = VectorAssembler(
    inputCols=["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"],
    outputCol="features")
output = assembler.transform(df2)
lr = LogisticRegression(labelCol="label",featuresCol="features")
fittedLR = lr.fit(output)
fittedLR.transform(output).select("label", "prediction").show()

# Good to stop SparkSession at the end of the application
session.stop()