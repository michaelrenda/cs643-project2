from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import SparkSession
import pyspark.sql as sql
from pyspark.sql.types import DoubleType
import os
import sys
import shutil
import numpy as np
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import RFormula
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import TrainValidationSplit
from pyspark.ml.pipeline import Transformer

def cleanCSVFile(inputPath):

    file1 = open(inputPath, 'r') 
    Lines = file1.readlines() 
    line = Lines[0]
    line2 = line.replace(';', ',').strip('\n')
    line3 = line2.replace('"', '')
    column_headers = line3.split(',')
    Lines.pop(0) 
    data_list = []
    for line in Lines:
        line2 = line.replace(';', ',')
        line3 = line2.replace('"', '').strip('\n')
        data_line = tuple(map(float, line3.split(',')))
        data_list.append(data_line)
    df = session.createDataFrame(data=data_list, schema = column_headers)
    df2 = df.withColumn("label", df["quality"].cast(IntegerType()))
    return df2


model_path = ''

if len(sys.argv) > 1:
    model_path = sys.argv[1]
 
# delete the previous model if it exists
try:
    shutil.rmtree(model_path + 'renda_model')
except OSError as e:
    pass


sys.stdout = open("train.txt", "w")

session = SparkSession.builder.appName("Train RF Model").getOrCreate()

# Reads a CSV file with header, stores it in a dataframe
dfTrain1 = cleanCSVFile('TrainingDataset.csv')
dfTest1 = cleanCSVFile('ValidationDataset.csv')

assembler = VectorAssembler(
    inputCols=["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"],
    outputCol="features")
rf = RandomForestClassifier(labelCol="label",featuresCol="features")

stages = [assembler, rf]
pipeline = Pipeline().setStages(stages)

params = ParamGridBuilder()\
    .addGrid(rf.numTrees, [int(x) for x in np.linspace(start = 10, stop = 50, num = 5)])\
    .addGrid(rf.featureSubsetStrategy, ["auto", "all", "sqrt", "log2"])\
    .addGrid(rf.maxDepth, [int(x) for x in np.linspace(start = 5, stop = 25, num = 5)])\
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

model = tvs.fit(dfTrain1).bestModel


# Make predictions on test data. model is the model with combination of parameters
# that performed best.
tester = model.transform(dfTest1)
tester.select("features", "label", "prediction").show()
accuracy = evaluator.evaluate(tester)
print("F1 statistic on test dataset: " + str(accuracy))
sys.stdout.close

model.write().overwrite().save(model_path + "renda_model")
# Good to stop SparkSession at the end of the application
session.stop()

