# 微博文本情绪分析及可视化

## 一、项目简介

​	微博情感分析项目从微博热搜中获取热搜中的每一条微博文本，在进行数据预处理之后，通过机器学习中的朴素贝叶斯算法和支持向量机算法训练两个数据分类模型，同时，训练一个文本特征提取模型，在将模型保存后，实时爬取微博中的微博文本，加载训练好的模型进行分类，将其积极消极概率值作为其对应情感分值（-1到1之间，越接近1越积极，越接近-1越消极），用kafka将每一条微博文本及对应的情感分值传到spark，写入文件保存。最终，搭建一个网站，以可视化的方式来展现分析结果，结果有：微博情绪分值的柱状图和情感分类的占比图。

## 二、项目开发环境

### 1.总体环境

​	项目的总体环境为集群，操作系统的centos7，jdk版本是jdk-8u212，python版本是Python-3.7.0，spark版本是spark-3.0.2-bin-hadoop3.2，scala版本是scala-2.12.12，zookeeper版本是apache-zookeeper-3.5.8，kafka版本是kafka_2.12-2.7.0。

​	集群为：hadoop131、hadoop132、hadoop133。搭建了zookeeper集群和kafka集群。

### 2.数据获取和处理模块

​	这一模块主要是用来爬取微博文本数据，并对文本数据做数据预处理，以达到文本数据可分析、可预测的状态。在这个模块中主要用到的语言是Python，所用到的组件（Python库）主要有Requests、Urllib、Beautifulsoup、jieba、Re等。

​	Requests是一个功能强大且易于使用的库，使得发送和处理 HTTP 请求变得简单而便捷，广泛应用于爬虫、API 调用、Web 开发等领域。在这个项目中是用来获取微博热搜和热搜话题中每一条微博的html源代码。

​	Urllib是 Python 内置的标准库，提供了一系列用于处理 URL 的模块。它们可以帮助开发者进行网络资源的访问、URL 解析、错误处理和机器人协议的解析等操作。

​	Beautifulsoup是一个 Python 第三方库，用于解析 HTML、XML 等文档，并提供了简单而直观的方法来遍历文档树、搜索特定标签和提取所需信息。在这个项目中主要是用来提取关键的微博文本信息，为后续数据建模和预测做准备。

​	Re是 Python 内置的正则表达式库，全称为 Regular Expression。正则表达式是一种强大的文本处理工具，它用于匹配、查找和编辑字符串中的模式。在这个项目中主要是用来对微博文本数据进行清洗，去除微博文本中用户名、链接等对数据分析有影响的文本。

​	Jieba是一个基于 Python 的中文分词工具，它可以将连续的中文文本切分成一个个有意义的词语。中文分词是自然语言处理的重要预处理步骤，对于中文文本的理解和处理具有重要意义。在这个项目中jieba分词主要是用来对微博文本进行分词，以便提取微博文本的特征，还有去除微博文本中的停用词，停用词就是在语义表达中毫无意义的介词、连词等，去除之后能更好的对微博文本情绪进行分类。

### 3.模型训练模块

​	这一模块主要是对获取到的已经预处理好的微博文本数据进行特征提取，并通过朴素贝叶斯算法和支持向量机算法SVM训练两个对微博文本的情绪进行分类的分类模型。所用到的语言是Python，组件（Python库）有sklearn和joblib。

​	Scikit-learn是一个基于Python的机器学习库，提供了许多常用的机器学习算法和工具，包括分类、回归、聚类、降维、模型选择和数据预处理等。它的优点是易于使用，提供了丰富的文档和示例，适用于各种应用场景。

​	joblib是一个用于Python的库，用于高效地缓存和序列化Python对象。在机器学习中，我们通常需要对模型进行训练和评估，这可能需要大量的计算资源和时间。joblib能够解决这个问题，通过将训练好的模型缓存到磁盘上，以便在以后可以轻松地加载和使用它们。

### 4.模型应用模块

​	这一模块主要是用来加载训练好的分类模型，获取新的微博文本，处理后对每一条微博进行预测分类，并由kafka将微博文本和每一条预测分类结果传给spark，将对数据进行聚合。语言用的是Python，组件有Kafka、Spark、json。

​	Kafka是一个高性能的分布式流处理平台，由Apache软件基金会开发并开源。它具有高吞吐量、可持久化存储和分布式处理等特点。在本次项目中Kafka主要是将分类模型预测过后的文本和分值以json的格式传给Spark进行聚合操作。

​	Spark是一个快速、通用的大数据处理引擎。此项目中Spark主要是用来对Kafka传来的json格式的数据进行结构化数据处理，并事件生成的时间为基准进行处理和窗口操作，最后生成可以被可视化展现的csv文件。

### 5.分析结果可视化模块

​	在本次项目中，这个模块用来对分析好的数据文件在网页上进行可视化展现。用到的语言是Python，所用到的组件有Flask，Pyecharts等。

​	Flask是一个轻量级的Web应用框架，它基于Python语言开发，使用简单且易于学习，同时提供了灵活的扩展能力。此次项目Flask主要是用来搭建一个用于展示可视化分析结果的前端页面，在这个页面中可以方便的展示通过数据挖掘建模后形成的数据分析结果。

​	Pyecharts是一个基于Python的开源数据可视化库，它利用Echarts.js来生成交互式的图表和可视化效果。在这个项目中Pyecharts主要是用来将分析后的csv文件画成图表。

## 三、项目流程图

![项目流程图](img/微博热搜情绪分析项目流程图.jpg)

## 四、项目功能实现

### 1.weibo_top_sentiment_SVM_bayes.py

​	清洗好并带有标签的微博文本数据集存在./dataset目录中。先使用pandas读取数据集文件并划分训练集和测试集，使用深度学习的pipeline模型将TFIDF模型和朴素贝叶斯算法连接，用训练集对该模型进行训练，将训练后的模型用joblib保存为model对象并输出训练报告，再使用深度学习的pipeline模型将TFIDF模型和支持向量机算法进行连接，用训练集训练后输出训练报告并保存为model对象

```
# 训练朴素贝叶斯分类模型和SVM支出向量机分类模型，并用深度学习模型框架分别用TF_IDF特征提取模型连接朴素贝叶斯算法分类模型和svm支持向量机分类模型
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

re_test_data = pd.read_csv('./dataset/re_sentiment_data.csv',encoding='utf_8_sig')
X = re_test_data['text']
y = re_test_data.label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11)

# 将TFIDF模型和朴素贝叶斯算法连接
TFIDF_NB_Sentiment_Model = Pipeline([
    ('TFIDF', TfidfVectorizer()),
    ('NB', MultinomialNB())
])
# 取八万条数据进行训练
nbm = TFIDF_NB_Sentiment_Model.fit(X_train[:80000],y_train[:80000])
joblib.dump(TFIDF_NB_Sentiment_Model, './model/tfidf_nb_sentiment.model')
nb_train_score = TFIDF_NB_Sentiment_Model.score(X_test,y_test)
y_pred = TFIDF_NB_Sentiment_Model.predict(X_test)
bayes_classifier_report = classification_report(y_test,y_pred)
print("朴素贝叶斯模型的精准率："+str(nb_train_score))
print(bayes_classifier_report)

TFIDF_SVM_Sentiment_Model = Pipeline([
    ('TFIDF', TfidfVectorizer()),
    ('SVM', SVC(C=0.95,kernel="linear",probability=True))
])
TFIDF_SVM_Sentiment_Model.fit(X_train[:30000],y_train[:30000])
joblib.dump(TFIDF_SVM_Sentiment_Model, './model/tfidf_svm_sentiment.model')
svm_test_score = TFIDF_SVM_Sentiment_Model.score(X_test,y_test)
y_pred = TFIDF_SVM_Sentiment_Model.predict(X_test)
svm_classifier_report = classification_report(y_test,y_pred)
print("支持向量机模型的精准率："+str(svm_test_score))
print(svm_classifier_report)

```



### 2.weibo_top_sentiment_predict.py

​	主要作用是加载训练好的朴素贝叶斯分类模型或者支持向量机分类模型，在爬取新的微博文本数据时直接对文本的情感分值和分类进行预测。

​	这一部分包含了文本数据清洗的功能，为了在爬取新文本数处理文本数据，主要包括去重、去噪、分词、去除停用词。

​	这一部分还内置了boson情感分值的分析功能。对boson文本计算情感分值的原理是：boson有一个非常丰富的情感词典，包含中文英文词语和词语对应的情感分值，负面情感用负的数值代替，积极情感用正的数值代替，要计算一个句子的情感得分即将这个句子中所有在情感词典汇总出现过的词语的分值累加起来，得到boson情感分值

```
import pickle
import re
import joblib
import jieba
import pandas as pd

# bosonnlp
# 基于波森情感词典计算情感值
class Boson_nlp:
    #创建对象时输入如boson情感词典路径作为参数
    def __init__(self,BosonNLP_sentiment_dict_path):
        self.BosonNLP_sentiment_dict = BosonNLP_sentiment_dict_path

    def getscore(self, text):
        if not isinstance(text, str):
            text = str(text)  # 将非字符串对象转换为字符串
        df = pd.read_table(self.BosonNLP_sentiment_dict, sep=" ", names=['key', 'score'])
        key = df['key'].values.tolist()
        score = df['score'].values.tolist()
        segs = jieba.lcut(text)  # 返回list
        score_list = [score[key.index(x)] for x in segs if (x in key)]
        return sum(score_list)

# 数据清洗去除文本中的用户名和链接
def data_clean(text):
    # 去除@用户名
    text = re.sub(r'@[\w]+', '', text)
    # 去除URL链接
    text = re.sub(r'http[s]?://\S+', '', text)
    # 替换#.......#
    text = re.sub(r'#.*?#', '', text)
    # 替换【......】
    text = re.sub(r'【.*?】', '', text)
    # 替换L后微博名称
    text = re.sub(r'L.*?频', '', text)
    # 替换展开
    text = re.sub(r'展.*?c', '', text)
    # 替换零宽度空白符
    text = re.sub("\u200b", "", text)

    return text


# 获取停用词列表
def get_custom_stopwords():
    with open('./dataset/stopwords.txt', encoding='utf-8') as f:
        stopwords = f.read()
    stopwords_list = stopwords.split('\n')
    custom_stopwords_list = [i for i in stopwords_list]
    with open('./dataset/中文停用词库.txt', encoding='utf-8') as f1:
        stopwords1 = [item.strip() for item in f1.readlines()]
    with open('./dataset/哈工大停用词表.txt', encoding='utf-8') as f2:
        stopwords2 = [item.strip() for item in f2.readlines()]
    with open('./dataset/四川大学机器智能实验室停用词库.txt', encoding='utf-8') as f3:
        stopwords3 = [item.strip() for item in f3.readlines()]
    stop_words_list = stopwords1 + stopwords2 + stopwords3 + custom_stopwords_list
    return stop_words_list


# 去除停用词方法
def remove_stropwords(mytext, cachedStopWords):
    return " ".join([word for word in mytext.split() if word not in cachedStopWords])


# 处理否定词不的句子
def Jieba_Intensify(text):
    word = re.search(r"不[\u4e00-\u9fa5 ]", text)
    if word != None:
        text = re.sub(r"(不 )|(不[\u4e00-\u9fa5]{1} )", word[0].strip(), text)
    return text

# 加载训练好的朴素贝叶斯分类模型和支持向量机分类模型，传入微博文本，判断句子消极还是积极
def get_sentiment_score(text):
    # 加载训练好的模型
    model = joblib.load('./model/tfidf_nb_sentiment.model')

    # model = joblib.load('./model/tfidf_svm_sentiment.model')
    # 获取停用词列表
    cached_StopWords = get_custom_stopwords()
    # 文本去除噪声
    cleaned_text = data_clean(text)
    # 去除停用词
    text = remove_stropwords(cleaned_text, cached_StopWords)
    # jieba分词
    seg_list = jieba.cut(text, cut_all=False)
    text = " ".join(seg_list)
    # 否定不处理
    text = Jieba_Intensify(text)
    y_pre = model.predict([text])
    proba = model.predict_proba([text])[0]
    if y_pre[0] == 1:
        return proba[1],y_pre[0]
    else:
        return -proba[0],y_pre[0]

if __name__ == '__main__':
    print(get_sentiment_score("本来想睡会睡不了泪泪泪工作"))

```



### 3.weibo_top_producer.py

​	主要功能是爬取微博热搜榜标题中的每一条微博，调用weibo_top_sentiment_predict的函数对文本的情绪分值做预测，将文本和分类封持久化成csv文件，创建kafka生产者，将文本和分值封装成json，每预测一条文本的分值，就推送给kafka的weibotop主题。

​	在爬取每一条微博时需要对html文档进行解析，提取要分析预测的文本。

```
import urllib
import json
from kafka import KafkaProducer
import requests
from bs4 import BeautifulSoup
import time
import urllib.parse
import pandas as pd
from weibo_top_sentiment_predict import Boson_nlp, get_sentiment_score
import datetime

bootstrap_servers = ['192.168.128.131:9092', '192.168.128.132:9092', '192.168.128.133:9092']
producer = KafkaProducer(bootstrap_servers=bootstrap_servers,
                         key_serializer=lambda k: json.dumps(k).encode(),
                         value_serializer=lambda v: json.dumps(v).encode())

stopwordsPath = './dataset/stopwords.txt'
request_headers = {
      'Host':'s.weibo.com',
      'Accept':'*/*',
      'Accept-Encoding':'gzip,deflate,br',
      'Accept-Language':'zh-CN,zh;q=0.9',
      'Referer':'https://s.weibo.com/',
      'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36',
      'cookie':'SUB=_2AkMSXtwvf8NxqwJRmPEXz2zqa45wwgzEieKkAi30JRMxHRl-yT9kqlEDtRB6Od7ywB0FDHt_kUsDrxNhHLiztDGKNxqN; SUBP=0033WrSXqPxfM72-Ws9jqgMF55529P9D9WhVUjSPSJFlwNA_J1Mo05jS; SINAGLOBAL=5939120240091.941.1694651160759; _s_tentry=-; Apache=321146689053.744.1698975827996; ULV=1698975828009:9:1:1:321146689053.744.1698975827996:1697191842098'
}

boson = Boson_nlp("./dataset/BosonNLP_sentiment_score.txt")

def get_top(producer):
    url = 'https://s.weibo.com/top/summary'
    response = requests.get(url,headers=request_headers)
    response.encoding = 'utf-8'
    print(response.status_code)
    soup = BeautifulSoup(response.text, 'lxml')
    data = soup.select('#pl_top_realtimehot table tbody tr td:nth-child(2)')
    num = 1
    hot_list = []

    for item in data:
        # print('================================')
        title = item.a.text
        host_score = item.span.text if item.span else ""
        t = urllib.parse.quote(title)
        href = "https://s.weibo.com/weibo?q=%23" + t + "%23&t=31&band_rank=1&Refer=top"
        if host_score:
            # print(f'{num} {title} hot: {host_score} url: {href}')
            weibo_list = get_weibo_list(href)
            for weibo_str in weibo_list:
                weibo = [weibo_str]
                hot_list.append(weibo)
            num += 1
    print(hot_list)
    classification_result_df = pd.DataFrame(columns=['text','classification'])
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    for item in hot_list:
        weibo_text = "".join(item)
        result = get_sentiment_score(weibo_text)
        sentiment_score = result[0]
        classification = result[1]
        new_row = {'text':weibo_text,'classification':classification}
        classification_result_df = pd.concat([classification_result_df,pd.DataFrame(new_row,index=[0])])
        print(weibo_text,sentiment_score)
        msg = {"title": weibo_text,"sentiment_score": sentiment_score}
        producer.send('weibotop', key='ff', value=json.dumps(msg))
    classification_result_df.reset_index(drop=True)
    time.sleep(60)
    classification_result_df.to_csv(f"./result/rate/weibo_classification_{current_time}.csv",encoding='utf-8',index=False,)

def get_weibo_list(url):
    response = requests.get(url, headers=request_headers)
    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.text, 'lxml')
    div_card_wrap = soup.find_all('div', attrs={'class': 'card-wrap', 'action-type': 'feed_list_item'})
    # print(div_card_wrap)
    weibo_list = []
    for each_div in div_card_wrap:
        div_card = each_div.find('div', attrs={'class': 'card'})
        div_card_feed = div_card.find('div', attrs={'class': 'card-feed'})
        div_content = div_card_feed.find('div', attrs={'class': 'content'})
        p_feed_list_content = div_content.find('p', attrs={'class': 'txt', 'node-type': 'feed_list_content'})
        content_text = p_feed_list_content.get_text()
        p_feed_list_content_full = div_content.find('p', attrs={'class': 'text', 'node-type': 'feed_list_content_full'})
        if p_feed_list_content_full:
            content_text = p_feed_list_content_full.get_text()
        weibo_list.append(content_text.strip())
    return weibo_list

while True:
    get_top(producer=producer)
    time.sleep(3600)

```



### 4.weibo_top_consumer.py

​	kafka消费者以结构化流的形式，就是正在扩充的dataframe，从kafka的weibotop主题中获取json，并且转换成为python得dataframe最终写成用于分析的csv文件。

```
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

```



### 5.weibo_top_visual_pyecharts.py

​	使用flask搭建一个网站，并用pyecharts画微博文本情绪分值为柱状图，和微博积极消极情绪占比饼图，在网页上可以访问到生成的页面。

​	需要使用pyecharts画一个柱状图，首先在./result/score目录下中读取数据，根据csv文件的文件名判断找出与当前时间最近的分析文件，text为微博文本，score为文本对应的情感分值（-1到1之间），画出的柱状图横轴为每一条微博文本，但是由于微博文本数量很多且文本字数很多，需要再横轴加一个缩放条，并将文本信息隐藏在横坐标的点上，在鼠标移到柱子上时展示完整的微博文本，纵轴为对应的情感分值为-1到1之间，大于0展示为红色，小于0的展示为蓝色。

​	饼图，首先在./result/rate目录下中读取数据，根据csv文件的文件名判断找出与当前时间最近的分析文件，text为微博文本，classification为文本对应的情感分类（0为消极情绪，1为积极情绪），使用pyecharts画饼图。

​	最后在index.html页面设置指向柱状图和饼图的链接。

```
import os
import glob
import datetime
from flask import *
from pyecharts import options as opts
from pyecharts.charts import Bar, Pie, Page
import pandas as pd

app = Flask(__name__)

@app.route("/")
def index():
    bar = get_data_bar()
    pie = get_data_pie()

    page_1 = Page()
    page_1.add(bar)

    page_2 = Page()
    page_2.add(pie)

    page_1.render('./static/sentiment_score_bar_page_1.html')
    page_2.render('./static/sentiment_ratio_pie_page_2.html')
    return render_template('index.html')


def get_data_bar():
    filename = find_latest_file(1)
    df = pd.read_csv(filename)
    # 从文件名中提取时间
    time_str = filename.split('_')[2].replace('-', ':')
    # 添加 x 轴和 y 轴数据
    x_data = df['text'].tolist()
    y_data = df['sentiment_score'].tolist()
    sentiment_score_bar = (
        Bar(init_opts=opts.InitOpts(width='100%', height='800px'))
        .add_xaxis(x_data)
        .add_yaxis("情感分值", y_data)
        .set_global_opts(
            title_opts=opts.TitleOpts(title="微博情感分值柱状图", subtitle=f"时间: {time_str}"),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(interval=0)),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="shadow"),
            datazoom_opts=[opts.DataZoomOpts(type_="slider", orient="horizontal")]
        )
        .set_series_opts(label_opts=None)
    )
    return sentiment_score_bar


def get_data_pie():
    file_name = find_latest_file(2)
    df = pd.read_csv(file_name)
    classification_counts = df['classification'].value_counts()
    index_mapping = {0: "消极", 1: "积极"}
    classification_series = classification_counts.rename(index_mapping)
    x = classification_series.index.tolist()
    y = classification_series.tolist()

    sentiment_ratio_pie = (
        Pie(init_opts=opts.InitOpts(theme='chalk',width='100%', height='800px'))
        .add(
            '情绪',
            # 只能传入列表
            list(zip(x, y)),
        )
        .set_colors(['red', 'blue'])
        .set_series_opts(
            # 设置标签
            label_opts=opts.LabelOpts(
                formatter='{b}:{c}，占{d}%'
            )
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title='微博情绪占比分析',
                subtitle=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
        )
    )
    return sentiment_ratio_pie


def find_latest_file(sign):
    folder_path = None
    if sign == 1:
        folder_path = "./result/score"
    elif sign == 2:
        folder_path = "./result/rate"
    else:
        print("file not found")
    current_time = datetime.datetime.now()
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    min_time_diff = float('inf')
    selected_file = None
    for file in csv_files:
        file_name = os.path.basename(file)
        time_str = file_name.split('_')[2].split('.')[0]
        date_format = "%Y-%m-%d-%H-%M-%S"
        file_time = datetime.datetime.strptime(time_str, date_format)
        time_diff = (current_time - file_time).total_seconds()
        if time_diff < min_time_diff:  # 选择时间差大于等于 0 的文件
            min_time_diff = time_diff
            selected_file = file

    return selected_file



if __name__ == '__main__':
    app.run(host='localhost', port=5000, processes=1)
```



## 五、项目结果

### 1.模型训练

```
python3 weibo_top_sentiment_SVM_bayes.py
```

![模型训练]([img/屏幕截图%2023-11-06%145407.png](https://github.com/FreakKJ/EmotionAnalysisForWeibo/blob/main/img/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202023-11-06%20145407.png))


支持向量机模型的精确率稍微好一些

### 2.生产者启动

```
python3 weibo_top_producer.py
```

![生产者启动1](img/屏幕截图2023-11-094012.png)

![生产者启动2](img/屏幕截图2023-11-094123.png)

![生产者启动3](img/屏幕截图2023-11-094142.png)

![生产者启动4](img/屏幕截图2023-11-094203.png)

![生产者启动5](img/屏幕截图2023-11-094224.png)

### 3.消费者启动

```
spark-submit --packages org.apache.spark:spark-streaming-kafka-0-10_2.12:3.0.1,org.apache.spark:spark-sql-kafka-0-10_2.12:3.0.1 weibo_top_consumer.py 192.168.128.131:9092,192.168.128.132:9092,192.168.128.133:9092 subscribe weibotop
```

![消费者启动1](img/屏幕截图2023-11-094153.png)

![消费者启动2](img/屏幕截图2023-11-094256.png)

已经生成了新的csv文件

![生成](img/屏幕截图2023-11-07094736.png)

### 4.flask服务器启动

```
python3 weibo_top_visual_pyecharts.py
```

首页

![首页](img/屏幕截图2023-11-07095640.png)

情感分值柱状图

![情感分值柱状图](img/屏幕截图2023-11-07095652.png)

缩放进度条放大后的

![情感分值柱状图1](img/屏幕截图2023-11-07095713.png)

![情感分值柱状图2](img/屏幕截图2023-11-07095729.png)

![情感分值柱状图3](img/屏幕截图2023-11-07095742.png)

情绪占比图

![情感分值柱状图3](img/屏幕截图2023-11-07095759.png)





#### 参考：

#### [基于kafka的微博情感分析与可视化实现 | 倬倬吃三碗 (zhuozhuo233.github.io)](https://zhuozhuo233.github.io/2021/11/05/基于kafka的微博情感分析与可视化实现/)

#### [基于支持向量机SVM和朴素贝叶斯NBM情感分析_python 支持向量机 情感分析_拼命_小李的博客-CSDN博客](https://blog.csdn.net/m0_43432638/article/details/122142472?ops_request_misc=%7B%22request%5Fid%22%3A%22169932563316800213067831%22%2C%22scm%22%3A%2220140713.130102334.pc%5Fall.%22%7D&request_id=169932563316800213067831&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-13-122142472-null-null.142^v96^pc_search_result_base2&utm_term=朴素贝叶斯微博情感分析&spm=1018.2226.3001.4187)
>>>>>>> 

