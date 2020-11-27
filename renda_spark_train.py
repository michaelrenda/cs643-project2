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

train_local = False

class LabelSetter(Transformer):
    # Label Setter herit of property of Transformer
    def __init__(self, inputCol='quality', outputCol='label'): 
        self.inputCol = inputCol
        self.outputCol = outputCol
    def _transform(self, df):
        return df.withColumn(self.outputCol, df[self.inputCol])

sys.stdout = open("test.txt", "w")

if train_local == True:
    # Creates a session on a local master
    session = SparkSession.builder.appName("CSV to Dataset").master("local[*]").getOrCreate()
    # Reads a CSV file with header, stores it in a dataframe
    dfTrain = session.read.csv(header=True, inferSchema=True, path='TrainingDataset.csv')
    dfTest = session.read.csv(header=True, inferSchema=True, path='ValidationDataset.csv')
else:
    session = SparkSession.builder.appName("CSV to Dataset").getOrCreate()
    dfTrain = session.read.csv(header=True, inferSchema=True, path='s3n://renda-spark-input/TrainingDataset.csv')
    dfTest = session.read.csv(header=True, inferSchema=True, path='s3n://renda-spark-input/ValidationDataset.csv')


assembler = VectorAssembler(
    inputCols=["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"],
    outputCol="features")
ls = LabelSetter("quality")
lr = LogisticRegression(labelCol="label",featuresCol="features", family="multinomial")

stages = [assembler, ls, lr]
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

model = tvs.fit(dfTrain)

# Make predictions on test data. model is the model with combination of parameters
# that performed best.
tester = model.transform(dfTest)
tester.select("features", "label", "prediction").show()
accuracy = evaluator.evaluate(tester)
print("F1 statistic on test dataset: " + str(accuracy))


# Good to stop SparkSession at the end of the application
session.stop()

sys.stdout.close