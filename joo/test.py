import numpy as np
import pickle

from konlpy.tag import Okt
from scipy.sparse import lil_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

def read_data(filename):
    f = open(filename, 'r', encoding='utf-8')
    datas = f.readlines()
    result = []
    for data in datas:
        result.append(data)
    return result

datas = read_data('../ratings_test.txt')[0]

# print(datas)

okt = Okt()


def tokenize(doc):
    return okt.pos(u'{}'.format(doc), norm=True, stem=True)

tok_data = tokenize(datas)

print(tok_data)