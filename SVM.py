from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LinearSVC
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics, RegressionMetrics
import pandas as pd

# setup
sc = SparkContext("local", "polifact")
sqlc = SQLContext(sc)

#training data
rdd = sc.textFile('/liar_dataset/train.tsv')
splitData1 = rdd.flatMap(lambda df1: df1.split('\n'))
splitData2 = splitData1.map(lambda sd: sd.split('\t'))
temp = splitData2.toDF()
trainingData = temp.toPandas()
training2 = temp

#testing data
rdd = sc.textFile('/liar_dataset/test.tsv')
splitData1 = rdd.flatMap(lambda df1: df1.split('\n'))
splitData2 = splitData1.map(lambda sd: sd.split('\t'))
temp = splitData2.toDF()
testingData = temp.toPandas()

# variables
subject = []
speaker = []
speaker_job = []
state = []
party = []
##########################################################
#functions
def change_label(col):
    for i in range(0, col.size):
        if col[i] == "true" or col[i] == "mostly-true" or col[i] == "half-true" or col[i] == "barely-true":
            col[i] = 0
        else: # false, and pants on fire
            col[i] = 1

def change_numeric(features, col):
    for i in range(0,col.size):
        if col[i] in features:
            col[i] = features.index(col[i])
        else:
            features.append(col[i])
            col[i] = features.index(col[i])

def count_to_int(col):
    for i in range(0,col.size):
        col[i] = int(col[i])

def clean_data(df):
    del df['_1']    #JSON id is not useful (i think)
    del df['_14']   #context can't be made numerical
    ########################################################
    change_label(df['_2'])   #label

    change_numeric(subject, df['_4'])       #subject
    change_numeric(speaker, df['_5'])       #speaker
    change_numeric(speaker_job, df['_6'])   #speaker job
    change_numeric(state, df['_7'])         #state
    change_numeric(party, df['_8'])         #party

    count_to_int(df['_9'])   #barely-true count
    count_to_int(df['_10'])  #false count
    count_to_int(df['_11'])  #half-true count
    count_to_int(df['_12'])  #mostly-true count
    count_to_int(df['_13'])  #pants-on-fire bount
    ##########################################################
    df.rename(columns={'_2':'label', '_3':'statement','_4':'subject', '_5':'speaker',
                        '_6':'speaker-job','_7':'state','_8':'party','_9':'barely-true',
                        '_10':'false', '_11':'half-true','_12':'mostly-true','_13':'pants-on-fire'}, 
                        inplace=True)
##########################################################
###################### Stats Stuff :) ####################
##########################################################
def stats_generator(predictionAndLabels):
    stats =[]
    metrics = BinaryClassificationMetrics(predictionAndLabels)
    metrics2 = MulticlassMetrics(predictionAndLabels)
    metrics3 = RegressionMetrics(predictionAndLabels)
    stats.append("Binary Classification Metrics")
    stats.append("Area under the precision-recall curve: %f" %metrics.areaUnderPR)
    stats.append("Area under the receiver operating characteristic (ROC) curve: %f" %metrics.areaUnderROC)
    stats.append("Regression Metrics")
    stats.append("Explained variance regression score: %f" %metrics3.explainedVariance)
    stats.append("Mean absolute error: %f" %metrics3.meanAbsoluteError)
    stats.append("Mean squared error: %f" %metrics3.meanSquaredError)
    stats.append("Square root of the mean squared error: %f" %metrics3.rootMeanSquaredError)
    stats.append("Multiclass Metrics")
    stats.append("Model accuracy: %f" %metrics2.accuracy)
    stats.append("Weighted false positive rate: %f" %metrics2.weightedFalsePositiveRate)
    stats.append("Weighted averaged precision:%f" %metrics2.weightedPrecision)
    stats.append("Weighted averaged recall: %f" %metrics2.weightedRecall)
    return stats

clean_data(trainingData)
clean_data(testingData)

#training data
svmtrain = trainingData.to_numpy().tolist()
for i in range(0,len(svmtrain)):
    svmtrain[i] = LabeledPoint(svmtrain[i][0], svmtrain[i][2:len(svmtrain)])

#train SVM model
svm = SVMWithSGD.train(sc.parallelize(svmtrain), iterations=10)

# test model
tDataLabel = []
tDatafeatures = []
for i in testingData.to_numpy().tolist():
    tDataLabel.append(i[0])
    tDatafeatures.append(i[2:])

pL = [] 
for j in range(0, len(tDataLabel)):
    pL.append((float(svm.predict(tDatafeatures[j])), float(tDataLabel[j])))

logisticRegressionStats = sc.parallelize(stats_generator(sc.parallelize(pL)))

logisticRegressionStats.saveAsTextFile("/results/SVM_stats_binary")
