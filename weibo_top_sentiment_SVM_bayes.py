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


