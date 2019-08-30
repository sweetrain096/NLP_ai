import pickle
import os
from threading import Thread
import sqlite3

import numpy as np
from konlpy.tag import Okt
from flask import Flask
from slack import WebClient
from slackeventsapi import SlackEventAdapter
from scipy.sparse import lil_matrix


# slack 연동 정보 입력 부분
# SLACK_TOKEN = os.getenv('SLACK_TOKEN')
# SLACK_SIGNING_SECRET = os.getenv('SLACK_SIGNING_SECRET')
SLACK_TOKEN = 'xoxb-507382705603-731558944624-2AnM7unjWKiirrXnWIV0kXOX'
SLACK_SIGNING_SECRET = 'c54cf581168e2786fa22dd60dc7b7b45'

app = Flask(__name__)

slack_events_adaptor = SlackEventAdapter(SLACK_SIGNING_SECRET, "/listening", app)
slack_web_client = WebClient(token=SLACK_TOKEN)

# Req 2-2-1. pickle로 저장된 model.clf 파일 불러오기

with open('model_add_knn.clf', 'rb') as f:
    model = pickle.load(f)

word_indices = model.get_word_indices()
naive = model.get_naive_model()
logi = model.get_logistic_model()
knn = model.get_k_neighbors_model()



# Req 2-2-2. 토큰화 및 one-hot 임베딩하는 전 처리

okt = Okt()


def tokenize(doc):
    tt = okt.pos(doc, norm=True, stem=True)
    return ['/'.join(t) for t in tt]


def preprocess(text):

    vocas = tokenize(text)
    X = [0] * (len(word_indices) + 1)

    for voca in vocas:
        indices = word_indices.get(voca)
        if indices:
            X[indices] = 1
    X = [X]
    return np.array(X)


# Req 2-2-3. 긍정 혹은 부정으로 분류


class Cf:

    def __init__(self, cf_model):
        self.cf_model = cf_model

    def predict(self, text):
        data = preprocess(text)
        result = self.cf_model.predict(data)[0]

        if result == 1:
            return '긍정'
        elif result == 0:
            return '부정'
        else:
            return '오류'


def classify(text):
    
    data = preprocess(text)
    result = naive.predict(data)[0]

    if result == 1:
        return '긍정'
    elif result == 0:
        return '부정'
    else:
        return '오류'


# Req 2-2-4. app.db 를 연동하여 웹에서 주고받는 데이터를 DB로 저장


def insert(text):
    conn = sqlite3.connect('app.db')
    c = conn.cursor()
    c.execute('INSERT INTO search_history(query) VALUES(?)', (text,))
    conn.commit()
    conn.close()


# 챗봇이 멘션을 받았을 경우
@slack_events_adaptor.on("app_mention")
def app_mentioned(event_data):
    channel = event_data["event"]["channel"]
    text = event_data["event"]["text"]

    text_list = text.split()
    idkey = text_list[0]
    classify_style = text_list[1]
    text_data = ' '.join(text_list[2:])

    if classify_style == 'naive':
        keywords = Cf(naive).predict(text_data) + ' with naive'
        insert(text_data)
    elif classify_style == 'logi':
        keywords = Cf(logi).predict(text_data) + ' with logi'
        insert(text_data)
    elif classify_style == 'knn':
        keywords = Cf(knn).predict(text_data) + ' with knn'
        insert(text_data)
    else:
        keywords = '분류이름을 맨 앞에 띄어쓰기로 구분해서 넣어주세요 [naive, logi, knn]'
        slack_web_client.chat_postMessage(
            channel=channel,
            text=keywords
        )
        keywords =  'ex) naive 이 영화 정말 재밌네요'

    slack_web_client.chat_postMessage(
        channel=channel,
        text=keywords
    )


@app.route("/", methods=["GET"])
def index():
    return "<h1>Server is ready.</h1>"


if __name__ == '__main__':
    app.run()
