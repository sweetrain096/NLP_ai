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
SLACK_TOKEN = os.getenv('SLACK_TOKEN')
SLACK_SIGNING_SECRET = os.getenv('SLACK_SIGNING_SECRET')

return_text = ""

app = Flask(__name__)

slack_events_adaptor = SlackEventAdapter(SLACK_SIGNING_SECRET, "/listening", app)
slack_web_client = WebClient(token=SLACK_TOKEN)

# Req 2-2-1. pickle로 저장된 model.clf 파일 불러오기

with open('model.clf', 'rb') as f:
    model = pickle.load(f)

word_indices = model.get_word_indices()
naive = model.get_naive_model()
logi = model.get_logistic_model()
knn = model.get_k_neighbors_model()


# Req 2-2-2. 토큰화 및 one-hot 임베딩하는 전 처리
def preprocess(slack_str):
    X_test = lil_matrix((1, len(word_indices) + 1), dtype=int)
    test_str = ' '.join(slack_str.split()[1:])
    print(test_str)
    okt = Okt()
    tokenize = okt.pos(test_str, norm=True, stem=True)
    test_tokens = ['/'.join(t) for t in tokenize]
    none_cnt = 0
    for token in test_tokens:
        indices = word_indices.get(token)
        if indices:
            X_test[0, indices] = 1
        else:
            none_cnt += 1

    if none_cnt / len(test_tokens) > 0.5:
        return 0

    return X_test[0]


# Req 2-2-3. 긍정 혹은 부정으로 분류
def classify(X_test):
    if type(X_test) == int:
        return "잘 모르겠는데요..."
    # print(X_test)
    result1 = naive.predict(X_test)
    result2 = logi.predict(X_test)
    result3 = knn.predict(X_test)
    # print(result1, result2)
    if result1 + result2 + result3 >= 2:
        return "긍정"
    else:
        return "부정"


# Req 2-2-4. app.db 를 연동하여 웹에서 주고받는 데이터를 DB로 저장
def insert(text):
    # app.db 파일을 연결
    conn = sqlite3.connect('app.db')
    # 사용할 수 있도록 설정(?)
    c = conn.cursor()
    # 쿼리문
    c.execute('INSERT INTO search_history(query) VALUES(?)', (text,))
    # 저장
    conn.commit()
    # app.db 파일 연결 해제
    conn.close()


# 슬랙 연동 안될 시 테스트
# print(classify(preprocess("<ㄴㅁㅅㄷㅈㅁ> 재미있어요")))

# 챗봇이 멘션을 받았을 경우
@slack_events_adaptor.on("app_mention")
def app_mentioned(event_data):
    global return_text
    channel = event_data["event"]["channel"]
    text = event_data["event"]["text"]
    if text != return_text:
        return_text = text
        reply = classify(preprocess(text))
        print(reply)
        slack_web_client.chat_postMessage(
            channel=channel,
            text=reply
        )


@app.route("/", methods=["GET"])
def index():
    return "<h1>Server is ready.</h1>"

if __name__ == '__main__':
    app.run()
