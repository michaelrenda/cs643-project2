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

train_local = True

sys.stdout = open("test.txt", "w")

if train_local == True:
    # Creates a session on a local master
    session = SparkSession.builder.appName("Train LR Model").master("local[*]").getOrCreate()
    f_path = ''
else:
    session = SparkSession.builder.appName("Train LR Model").getOrCreate()
    f_path = 's3n://renda-spark-input/'

# Reads a CSV file with header, stores it in a dataframe
dfTrain = session.read.csv(header=True, inferSchema=True, path=f_path + 'TrainingDataset.csv')
dfTest = session.read.csv(header=True, inferSchema=True, path=f_path + 'ValidationDataset.csv')
logisticRegressionModelLoaded = TrainValidationSplitModel.load(f_path + "renda_model")

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