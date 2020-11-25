from pyspark.sql import SparkSession
import pyspark.sql as sql
from pyspark.sql.types import DoubleType
import os
from pyspark.ml.feature import RFormula
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

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


# Train a DecisionTree model.
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=1000)

# Chain indexers and tree in a Pipeline
pipeline = Pipeline(stages=[rf])

# Train model.  This also runs the indexers.
model = pipeline.fit(dfTrain3)

# Make predictions.
predictions = model.transform(dfTest3)

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
