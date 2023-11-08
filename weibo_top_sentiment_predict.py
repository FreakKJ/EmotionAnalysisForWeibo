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




