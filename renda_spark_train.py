from pyspark.sql import SparkSession
import pyspark.sql as sql
from pyspark.sql.types import DoubleType
import os
import sys
from pyspark.ml.feature import RFormula
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import TrainValidationSplit
from pyspark.ml.pipeline import Transformer

def cleanCSVFile(inputPath, outputPath):
    file1 = open(inputPath, 'r') 
    file2 = open(outputPath, 'w')
    Lines = file1.readlines() 
  
    for line in Lines: 
        line2 = line.replace(';', ',')
        line3 = line2.replace('"', '')
        file2.write(line3)
    file1.close()
    file2.close()
    return outputPath


f_path = ''
model_path = ''

if len(sys.argv) > 1:
    f_path = sys.argv[1]
    if len(sys.argv) > 2:
        model_path = sys.argv[2]

sys.stdout = open("train.txt", "w")

session = SparkSession.builder.appName("Train LR Model").getOrCreate()

# Reads a CSV file with header, stores it in a dataframe
training_file = cleanCSVFile(f_path + 'TrainingDataset.csv', 'training.csv')
testing_file = cleanCSVFile(f_path + 'ValidationDataset.csv', 'testing.csv')
dfTrain = session.read.csv(header=True, inferSchema=True, path=training_file)
dfTest = session.read.csv(header=True, inferSchema=True, path=testing_file)

dfTrain1 = dfTrain.withColumn("label", dfTrain["quality"])
dfTest1 = dfTest.withColumn("label", dfTest["quality"])
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

