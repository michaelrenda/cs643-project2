from pyspark.sql import SparkSession
import pyspark.sql as sql
from pyspark.sql.types import DoubleType
from pyspark.sql.types import IntegerType
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

f_path = 'ValidationDataset.csv'
model_path = 'renda_model'

if len(sys.argv) > 1:
    f_path = sys.argv[1]
if len(sys.argv) > 2:
    model_path = sys.argv[2]


session = SparkSession.builder.appName("Test LR Model").getOrCreate()

# Reads a CSV file with header, stores it in a dataframe
dfTest1 = cleanCSVFile(f_path)
logisticRegressionModelLoaded = PipelineModel.load(model_path)

evaluator = MulticlassClassificationEvaluator()\
  .setMetricName("f1")\
  .setPredictionCol("prediction")\
  .setLabelCol("label")

tester = logisticRegressionModelLoaded.transform(dfTest1)
row_count = tester.count()
print("predicting values for: " + f_path)
tester.select("features", "label", "prediction").show(row_count)
accuracy = evaluator.evaluate(tester)
print("F1 statistic on test dataset: " + str(accuracy))

# Good to stop SparkSession at the end of the application
session.stop()
