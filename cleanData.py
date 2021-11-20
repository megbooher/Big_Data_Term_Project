from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
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
        if col[i] == "true":
            col[i] = 0
        elif col[i] == "mostly-true":
            col[i] = 1
        elif col[i] == "half-true":
            col[i] = 2
        elif col[i] == "barely-true":
            col[i] = 3
        elif col[i] == "false":
            col[i] = 4
        else:
            col[i] = 5

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

clean_data(trainingData)
clean_data(testingData)
