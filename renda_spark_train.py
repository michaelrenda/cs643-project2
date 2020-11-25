from pyspark.sql import SparkSession
import pyspark.sql as sql
from pyspark.sql.types import DoubleType
import os
from pyspark.ml.feature import RFormula
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import TrainValidationSplit

# Creates a session on a local master
session = SparkSession.builder.appName("CSV to Dataset").master("local[*]").getOrCreate()

# Reads a CSV file with header, called books.csv, stores it in a dataframe
dfTrain = session.read.csv(header=True, inferSchema=True, path='TrainingDataset.csv')
dfTrain2 = dfTrain.withColumn("label", dfTrain["quality"].cast(DoubleType()))
assembler = VectorAssembler(
    inputCols=["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"],
    outputCol="features")
dfTrain3 = assembler.transform(dfTrain2)

dfTest = session.read.csv(header=True, inferSchema=True, path='ValidationDataset.csv')
dfTest2 = dfTest.withColumn("label", dfTest["quality"].cast(DoubleType()))
assembler = VectorAssembler(
    inputCols=["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"],
    outputCol="features")
dfTest3 = assembler.transform(dfTest2)

lr = LogisticRegression(labelCol="label",featuresCol="features", maxIter=100, regParam=0.02, elasticNetParam=0.8)
stages = [lr]
pipeline = Pipeline().setStages(stages)

params = ParamGridBuilder()\
    .addGrid(lr.elasticNetParam, [0.0, 0.25, 0.5, .075, 1.0])\
    .addGrid(lr.regParam, [0.1, 0.5, 1.0, 1.5, 2.0])\
    .addGrid(lr.maxIter, [100])\
    .build()

evaluator = MulticlassClassificationEvaluator()\
  .setMetricName("f1")\
  .setPredictionCol("prediction")\
  .setLabelCol("label")


tvs = TrainValidationSplit()\
  .setTrainRatio(1.0)\
  .setEstimatorParamMaps(params)\
  .setEstimator(pipeline)\
  .setEvaluator(evaluator)


# COMMAND ----------

model = tvs.fit(dfTrain3)

# Make predictions on test data. model is the model with combination of parameters
# that performed best.
tester = model.transform(dfTest3)
tester.select("features", "label", "prediction").show()
accuracy = evaluator.evaluate(tester)
print(accuracy)


# Good to stop SparkSession at the end of the application
session.stop()