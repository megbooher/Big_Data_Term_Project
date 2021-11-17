from pyspark import SparkConf, SparkContext
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint

sc = SparkContext("local", "polifact")

#training data
df1 = c.textFile('/liar_dataset/train.tsv',header)
trainingData = df1.flatMap(lambda df1: df1.split('\n'))

#testing data
df2 = sc.textFile('/liar_dataset/test.tsv')
testingData = df2.flatMap(lambda df2: df2.split('\n'))

# need to get trainingData and testingData into LabeledPoint RDD
# model = RandomForest.trainClassifier(trainingData, numClasses=5, 
#     categoricalFeaturesInfo={}, numTrees=3, 
#     featureSubsetStrategy="auto", impurity='gini', 
#     maxDepth=4, maxBins=32)
