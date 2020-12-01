from pyspark.sql import SparkSession
import pyspark.sql as sql
from pyspark.sql.types import DoubleType
import os
import sys
import shutil
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import RFormula
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
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

# delete the previous model if it exists
try:
    shutil.rmtree('renda_model')
except OSError as e:
    pass


if len(sys.argv) > 1:
    model_path = sys.argv[1]
 
# delete the previous model if it exists
try:
    shutil.rmtree(model_path + 'renda_model')
except OSError as e:
    pass


sys.stdout = open("train.txt", "w")

session = SparkSession.builder.appName("Train LR Model").getOrCreate()

# Reads a CSV file with header, stores it in a dataframe
dfTrain1 = cleanCSVFile('TrainingDataset.csv')
dfTest1 = cleanCSVFile('ValidationDataset.csv')

assembler = VectorAssembler(
    inputCols=["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"],
    outputCol="features")
lr = LogisticRegression(labelCol="label",featuresCol="features", family="multinomial")

stages = [assembler, lr]
pipeline = Pipeline().setStages(stages)

params = ParamGridBuilder()\
    .addGrid(lr.elasticNetParam, [0.0, 0.25, 0.5, .075, 1.0])\
    .addGrid(lr.regParam, [0.01, 0.1, 0.5, 1.0, 1.5, 2.0])\
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

