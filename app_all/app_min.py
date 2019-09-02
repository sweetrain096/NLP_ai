import pickle
from threading import Thread
import sqlite3
from model import Model

import numpy as np
from konlpy.tag import Okt
from flask import Flask
from slack import WebClient
from slackeventsapi import SlackEventAdapter
from scipy.sparse import lil_matrix

# slack 연동 정보 입력 부분
SLACK_TOKEN = "xoxb-718907786578-720177174562-oKkmHrXJhhDyxatiVMO3A9pK"
SLACK_SIGNING_SECRET = "fa3d6193e36d26163abc90a4507ceec8"

app = Flask(__name__)

slack_events_adaptor = SlackEventAdapter(SLACK_SIGNING_SECRET, "/listening", app)
slack_web_client = WebClient(token=SLACK_TOKEN)

# Req 2-2-1. pickle로 저장된 model.clf 파일 불러오기
with open('model.clf', 'rb') as f:
    pickle_obj = pickle.load(f)

word_indices = pickle_obj.get_word_indices()
clf = pickle_obj.get_naive_model()
clf2 = pickle_obj.get_logistic_model()

# Req 2-2-2. 토큰화 및 one-hot 임베딩하는 전 처리
okt = Okt()

def tokenize(doc):
    tt = okt.pos(doc, norm=True, stem=True)
    return ['/'.join(t) for t in tt]

def preprocess(sentence):
    words = tokenize(sentence)
    X = [0] * (len(word_indices) + 1)

    for word in words:
        indices = word_indices.get(word)
        if indices:
            X[indices] = 1

    X = [X]

    return np.array(X)

# Req 2-2-3. 긍정 혹은 부정으로 분류
def classify(sentence):
    data = preprocess(sentence)
    result = clf.predict(data)

    if result == 0:
        return '부정'

    elif result == 1:
        return '긍정'

    else:
        return '오류'

# Req 2-2-4. app.db 를 연동하여 웹에서 주고받는 데이터를 DB로 저장
def saveDB(db):
    conn = sqlite3.connect('app.db')
    cur = conn.cursor()

    cur.execute('INSERT INTO search_history (query) VALUES (?)', (db,))

    conn.commit()
    conn.close()

# 챗봇이 멘션을 받았을 경우
@slack_events_adaptor.on("app_mention")
def app_mentioned(event_data):
    channel = event_data["event"]["channel"]
    text = event_data["event"]["text"]

    keywords = classify(text)

    saveDB(text)
    slack_web_client.chat_postMessage(
        channel = channel,
        text=keywords
    )

@app.route("/", methods=["GET"])
def index():
    return "<h1>Server is ready.</h1>"


if __name__ == '__main__':
    app.run()
