import json
import sys
import time
import datetime
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType, StringType


@udf(returnType=StringType())
def gettitle(column):
    jsonobject = json.loads(column)
    jsonobject = json.loads(jsonobject)
    if "title" in jsonobject:
        return str(jsonobject['title'])
    return ""


@udf(returnType=DoubleType())
def getscore(column):
    jsonobject = json.loads(column)
    jsonobject = json.loads(jsonobject)
    if "sentiment_score" in jsonobject:
        return float(jsonobject['sentiment_score'])
    return 0.0


"""将 DataFrame 写入 CSV 文件"""
output_path = "./result/score"
processed_df = None
def merge_df_to_csv(batch_df, batch_id):
    global processed_df
    pandas_df = batch_df.toPandas()
    if processed_df is None:
        processed_df = pandas_df
    else:
        processed_df = pd.concat([processed_df, pandas_df])
    if processed_df.index.size >= 500:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        output_file = f"weibo_sentiment_{current_time}.csv"
        processed_df.to_csv(output_path + "/" + output_file, index=False)
        processed_df = None


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(-1)
    spark = SparkSession \
        .builder \
        .config("spark.pandas.version","1.1.5") \
        .appName("WeiboSpark") \
        .getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("ERROR")
    bootstrapServers = sys.argv[1]
    subscribeType = sys.argv[2]
    topics = sys.argv[3]
    lines = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", bootstrapServers) \
        .option(subscribeType, topics) \
        .option("failOnDataLoss", "false") \
        .load()
    kafka_value_tb = lines.selectExpr("CAST(value AS STRING) as json")
    weibo_table = kafka_value_tb.select(gettitle(col("json")).alias("text"),
                                        getscore(col("json")).alias("sentiment_score"))
    stat_avg_query = weibo_table.writeStream \
        .option("checkpointLocation", "/home/atguigu/checkpoint") \
        .option("header", "true") \
        .option("mode", "append") \
        .foreachBatch(merge_df_to_csv) \
        .start() \

    try:
        i = 1
        while True:
            print(stat_avg_query.status)
            time.sleep(10)
            i = i + 1
    except KeyboardInterrupt:
        print("process interrupted")

    stat_avg_query.awaitTermination()
