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


