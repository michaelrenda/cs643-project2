from pyspark.sql import SparkSession
import pyspark.sql as sql
from pyspark.sql.types import DoubleType
import os
import sys
from pyspark.ml.feature import RFormula
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import TrainValidationSplit
from pyspark.ml.pipeline import Transformer

run_local = True

class LabelSetter(Transformer):
    # Label Setter herit of property of Transformer
    def __init__(self, inputCol='quality', outputCol='label'): 
        self.inputCol = inputCol
        self.outputCol = outputCol
    def _transform(self, df):
        return df.withColumn(self.outputCol, df[self.inputCol])

sys.stdout = open("test.txt", "w")

if run_local == True:
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

# Train a DecisionTree model.
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=1000)

# Chain indexers and tree in a Pipeline
pipeline = Pipeline(stages=[assembler, ls, rf])

# Train model.  This also runs the indexers.
model = pipeline.fit(dfTrain)

# Make predictions.
predictions = model.transform(dfTest)

# Select example rows to display.
predictions.select("prediction", "label", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1")
accuracy = evaluator.evaluate(predictions)
print(accuracy)

treeModel = model.stages[0]
# summary only
print(treeModel)
# Good to stop SparkSession at the end of the application
session.stop()

sys.stdout.close