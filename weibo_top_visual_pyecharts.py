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