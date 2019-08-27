import numpy as np
import pickle

from konlpy.tag import Okt
from scipy.sparse import lil_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

"""
Req 1-1-1. 데이터 읽기
read_data(): 데이터를 읽어서 저장하는 함수
"""

def read_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = [line.split('\t') for line in f.read().splitlines()][1:]
    return data

"""
Req 1-1-2. 토큰화 함수
tokenize(): 텍스트 데이터를 받아 KoNLPy의 okt 형태소 분석기로 토크나이징
"""

pos_tagger = Okt()

def tokenize(doc):
    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]

"""
데이터 전 처리
"""

# train, test 데이터 읽기
train_data = read_data('ratings_train.txt')
test_data = read_data('ratings_test.txt')

# Req 1-1-2. 문장 데이터 토큰화
# train_docs, test_docs : 토큰화된 트레이닝, 테스트  문장에 label 정보를 추가한 list
train_docs = [(tokenize(row[1]), row[2]) for row in train_data]
test_docs = [(tokenize(row[1]), row[2]) for row in test_data]

# Req 1-1-3. word_indices 초기화
word_indices = {}

# Req 1-1-3. word_indices 채우기
for vocas in train_docs:
    for voca in vocas[0]:
        text = voca.split('/')[0]
        if text not in word_indices:
            word_indices[text] = len(word_indices)

# Req 1-1-4. sparse matrix 초기화
# X: train feature data
# X_test: test feature data
X = lil_matrix(len(train_docs), len(word_indices))
X_test = lil_matrix(len(test_docs), len(word_indices))

# 평점 label 데이터가 저장될 Y 행렬 초기화
# Y: train data label
# Y_test: test data label
Y = np.zeros(len(train_docs))
Y_test = np.zeros(len(test_docs))

# Req 1-1-5. one-hot 임베딩
# X,Y 벡터값 채우기

def one_hot_encoding(vocas, word_indice):
    one_hot_vector = [0] * (len(word_indices))
    for (idx, voca) in enumerate(vocas[0]):
        text = voca.split('/')[0]
        index = word_indices.get(text)
        if index is not None:
            one_hot_vector[index] = 1
    return one_hot_vector

for (idx1, vocas) in enumerate(train_docs):
    X[idx1] = one_hot_encoding(vocas, word_indices)

for (idx1, vocas) in enumerate(test_docs):
    X_test[idx1] = one_hot_encoding(vocas, word_indices)
​
for (idx1, vocas) in enumerate(train_docs):
     Y[idx1] = vocas[1]

for (idx1, vocas) in enumerate(test_docs):
     Y_test[idx1] = vocas[1]