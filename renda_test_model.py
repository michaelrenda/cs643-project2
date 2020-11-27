from pyspark.sql import SparkSession
import pyspark.sql as sql
from pyspark.sql.types import DoubleType
import os
import sys
from pyspark.ml.feature import RFormula
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import TrainValidationSplit
from pyspark.ml.tuning import TrainValidationSplitModel
from pyspark.ml.pipeline import Transformer
from pyspark.ml import PipelineModel


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


sys.stdout = open("test.txt", "w")

f_path = ''
model_path = ''

if len(sys.argv) > 1:
    f_path = sys.argv[1]
    if len(sys.argv) > 2:
        model_path = sys.argv[2]

session = SparkSession.builder.appName("Train LR Model").getOrCreate()

# Reads a CSV file with header, stores it in a dataframe
testing_file = cleanCSVFile(f_path + 'ValidationDataset.csv', 'testing.csv')
dfTest = session.read.csv(header=True, inferSchema=True, path=testing_file)
logisticRegressionModelLoaded = PipelineModel.load(model_path + "renda_model")

dfTest1 = dfTest.withColumn("label", dfTest["quality"])


evaluator = MulticlassClassificationEvaluator()\
  .setMetricName("f1")\
  .setPredictionCol("prediction")\
  .setLabelCol("label")

tester = logisticRegressionModelLoaded.transform(dfTest1)
row_count = tester.count()
tester.select("features", "label", "prediction").show(row_count)
accuracy = evaluator.evaluate(tester)
print("F1 statistic on test dataset: " + str(accuracy))

# Good to stop SparkSession at the end of the application
session.stop()

sys.stdout.close