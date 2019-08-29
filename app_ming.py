import os
import pickle
from threading import Thread
import sqlite3
# from model import Model

import numpy as np
from konlpy.tag import Okt
from flask import Flask
from slack import WebClient
from slackeventsapi import SlackEventAdapter
from scipy.sparse import lil_matrix


# slack 연동 정보 입력 부분
SLACK_TOKEN = os.environ['SLACK_TOKEN']
SLACK_SIGNING_SECRET = os.environ['SLACK_SIGNING_SECRET']
print(SLACK_TOKEN)
print(SLACK_SIGNING_SECRET)

app = Flask(__name__)

slack_events_adaptor = SlackEventAdapter(SLACK_SIGNING_SECRET, "/listening", app)
slack_web_client = WebClient(token=SLACK_TOKEN)

# Req 2-2-1. pickle로 저장된 model.clf 파일 불러오기
with open('model.clf', 'rb') as f:
    pickle_obj = pickle.load(f)
    word_indices = pickle_obj.get_word_indices()
    # print(word_indices)
    clf = pickle_obj.get_naive_model()
    clf2 = pickle_obj.get_logistic_model()

# Req 2-2-2. 토큰화 및 one-hot 임베딩하는 전 처리
# input : 영화평 문장
# output : 전 처리된 sparse matrix
text = '핵노잼'
def preprocess(text):
    okt = Okt()
    token = okt.pos(text)
    X = lil_matrix((1, len(word_indices)+1))
    # lil_matrix는 선언 할 때 괄호 두개, 1행짜리면 뒤에 , 1 붙여주기
    print(X)
    for voca in token:
        indices = word_indices.get('/'.join(voca))
        if indices:
            X[0, indices] = 1
    return X
print(preprocess(text))

# Req 2-2-3. 긍정 혹은 부정으로 분류
# input : 전 처리된 sparse matrix
# output : 부정적 리뷰 or 긍적적 리뷰(다양한 방식으로 분류 값 사용 가능)
def classify(X):
    if clf.predict(X):
        return '긍정'
    return '부정'

X = preprocess(text)
# print('X', X)
print(classify(X))
'''
# Req 2-2-4. app.db 를 연동하여 웹에서 주고받는 데이터를 DB로 저장
    

# 챗봇이 멘션을 받았을 경우
@slack_events_adaptor.on("app_mention")
def app_mentioned(event_data):
    channel = event_data["event"]["channel"]
    text = event_data["event"]["text"]


@app.route("/", methods=["GET"])
def index():
    return "<h1>Server is ready.</h1>"

if __name__ == '__main__':
    app.run()
'''