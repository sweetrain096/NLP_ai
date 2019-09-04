import numpy as np
import pickle
from model import Model

from konlpy.tag import Okt
from scipy.sparse import lil_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

"""
Req 1-1-1. 데이터 읽기
read_data(): 데이터를 읽어서 저장하는 함수
"""


def read_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        datas = [line.split('\t') for line in f.read().splitlines()]
        datas = datas[1:]
    return datas

"""
Req 1-1-2. 토큰화 함수
tokenize(): 텍스트 데이터를 받아 KoNLPy의 okt 형태소 분석기로 토크나이징
"""
okt = Okt()


def tokenize(doc):
    tt = okt.pos(doc, norm=True, stem=True)
    return ['/'.join(t) for t in tt]

"""
데이터 전 처리
"""

# train, test 데이터 읽기
train_data = read_data('ratings_train.txt')[:5]
test_data = read_data('ratings_test.txt')[:5]


# Req 1-1-2. 문장 데이터 토큰화
# train_docs, test_docs : 토큰화된 트레이닝, 테스트  문장에 label 정보를 추가한 list


train_docs = [(tokenize(i[1]), i[2]) for i in train_data]
test_docs = [(tokenize(i[1]), i[2]) for i in test_data]

# Req 1-1-3. word_indices 초기화
word_indices = {}

# Req 1-1-3. word_indices 채우기
# 기존
# for n_data in train_docs:
#     # 품사까지 dict화
#     for cnt in n_data[0]:
#         if not (word_indices.get(cnt)):
#             word_indices[cnt] = len(word_indices) + 1
# [index, 부정, 긍정]
for n_data in train_docs:
    # 품사까지 dict화
    for cnt in n_data[0]:
        if not (word_indices.get(cnt)):
            word_indices[cnt] = len(word_indices) + 1


    # 문자만 dict화
    # n_data = n_data.split('/')[0]
    # if not (word_indices.get(n_data)):
    #     word_indices[n_data] = len(word_indices) + 1
print(word_indices)

# Req 1-1-4. sparse matrix(희소행렬 = 거의 0으로 채워지고 몇개의 값만 값이 존재) 초기화
# X: train feature data
# X_test: test feature data
X = lil_matrix((len(train_data), len(word_indices) + 1))
X_test = lil_matrix((len(test_data), len(word_indices) + 1))


# 평점 label 데이터가 저장될 Y 행렬 초기화
# Y: train data label
# Y_test: test data label
Y = np.zeros((len(train_data)))
Y_test = np.zeros(((len(test_data))))

# Req 1-1-5. one-hot 임베딩
# X,Y 벡터값 채우기
for n in range(len(train_docs)):
    for token in train_docs[n][0]:
        indices = word_indices.get(token)
        if indices:
            X[n, indices] = 1
    Y[n] = train_docs[n][1]

for n in range(len(test_docs)):
    for token in test_docs[n][0]:
        indices = word_indices.get(token)
        if indices:
            X_test[n, indices] = 1
    Y_test[n] = test_docs[n][1]
'''
"""
트레이닝 파트
clf  <- Naive baysian mdoel
clf2 <- Logistic regresion model
"""

# Req 1-2-1. Naive bayes model 학습
clf = MultinomialNB()
clf.fit(X, Y)

# Req 1-2-2. Logistic regression model 학습
clf2 = LogisticRegression()
clf2.fit(X, Y)

# Req 2-3-3. 새로운 학습 모델 사용
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X, Y)


"""
테스트 파트
"""

# Req 1-3-1. 문장 데이터에 따른 예측된 분류값 출력
print("Naive bayesian classifier example result: {}, {}".format(test_data[3][1], clf.predict(X_test[3])))
print("Logistic regression exampleresult: {}, {}".format(test_data[3][1], clf2.predict(X_test[3])))
print("K Neighbors classifier example result: {}, {}".format(test_data[3][1], knn.predict(X_test[3])))

# Req 1-3-2. 정확도 출력
print("Naive bayesian classifier accuracy: {}".format(clf.score(X_test, Y_test)))
print("Logistic regression accuracy: {}".format(clf2.score(X_test, Y_test)))
print("K Neighbors classifier accuracy: {}".format(knn.score(X_test, Y_test)))


"""
데이터 저장 파트
"""

# Req 1-4. pickle로 학습된 모델 데이터 저장

model = Model()

model.set_naive_model(clf)
model.set_logistic_model(clf2)
model.set_k_neighbors_model(knn)
model.set_word_indices(word_indices)

with open('model_add_knn.clf', 'wb') as f:
   pickle.dump(model, f)
'''
# Naive bayes classifier algorithm part
# 아래의 코드는 심화 과정이기에 사용하지 않는다면 주석 처리하고 실행합니다.

"""
Naive_Bayes_Classifier 알고리즘 클래스입니다.
"""

class Naive_Bayes_Classifier(object):
    # def __init__(self):
    #     self.word_probs = []
    #
    # def word_probabilities(self, counts, total_class0, total_class1, k):
    #     # 단어의 빈도수를 [단어, p(w|부정), p(w|긍정)] 형태로 반환
    #     return [(w,
    #              (class0 + k) / (total_class0 + 2 * k),
    #              (class1 + k) / (total_class1 + 2 * k))
    #             for w, [index, class0, class1] in counts.items()]
    """
    Req 3-1-1.
    log_likelihoods_naivebayes():
    feature 데이터를 받아 label(class)값에 해당되는 likelihood 값들을
    naive한 방식으로 구하고 그 값의 log값을 리턴
    """
    
    def log_likelihoods_naivebayes(self, feature_vector, Class):
        log_likelihood = 0.0
        print(feature_vector)
        if Class == 0:
            for feature_index in range(len(feature_vector)):
                if feature_vector[feature_index] == 1: #feature present
                    log_likelihood += None
                elif feature_vector[feature_index] == 0: #feature absent
                    log_likelihood += None
        elif Class == 1:
            for feature_index in range(len(feature_vector)):
                if feature_vector[feature_index] == 1:
                    log_likelihood += None
                elif feature_vector[feature_index] == 0:
                    log_likelihood += None
                
        return None

    """
    Req 3-1-2.
    class_posteriors():
    feature 데이터를 받아 label(class)값에 해당되는 posterior 값들을
    구하고 그 값의 log값을 리턴
    """
    
    def class_posteriors(self, feature_vector):
        log_likelihood_0 = self.log_likelihoods_naivebayes(feature_vector, Class = 0)
        log_likelihood_1 = self.log_likelihoods_naivebayes(feature_vector, Class = 1)

        log_posterior_0 = None
        log_posterior_1 = None

        return None

    """
    Req 3-1-3.
    classify():
    feature 데이터에 해당되는 posterir값들(class 개수)을 불러와 비교하여
    더 높은 확률을 갖는 class를 리턴
    """    

    def classify(self, feature_vector):
        return None

    """
    Req 3-1-4.
    train():
    트레이닝 데이터를 받아 학습하는 함수
    학습 후, 각 class에 해당하는 prior값과 likelihood값을 업데이트

    알고리즘 구성
    1) 가중치 값인 beta_x_i, beta_c_i 초기화
    2) Y label 데이터 reshape
    3) 가중치 업데이트 과정 (iters번 반복) 
    3-1) prediction 함수를 사용하여 error 계산
    3-2) gadient_beta 함수를 사용하여 가중치 값 업데이트
    4) 최적화 된 가중치 값들 리턴
       self.beta_x, self.beta_c
    """

    def train(self, X, Y):
        # label 0에 해당되는 데이터의 개수 값(num_0) 초기화
        num_0 = 0

        num_1 = 0

        '''
        # 단어의 부정/긍정 확률 계산
        word_probs = self.word_probabilities(word_indices,
                                                  num_0,
                                                  num_1,
                                             smoothing)
        print(word_probs)
        '''
        # label 0에 해당되는 각 feature 성분의 개수값(num_token_0) 초기화
        num_token_0 = np.zeros((1, X.shape[1]))
        # label 1에 해당되는 각 feature 성분의 개수값(num_token_1) 초기화 
        num_token_1 = np.zeros((1, X.shape[1]))

        # 데이터의 num_0,num_1,num_token_0,num_token_1 값 계산     
        for i in range(X.shape[0]):
            if (Y[i] == 0):
                num_0 += 1

                num_token_0 += X[i]

            if (Y[i] == 1):
                num_1 += 1
                num_token_1 += X[i]

        # Req 3-1-7. smoothing 조절
        # likelihood 확률이 0값을 갖는것을 피하기 위하여 smoothing 값 적용
        smoothing = 0.5

        # smoothing을 사용하여 각 클래스에 해당되는 likelihood값 계산
        # likelihood.shape = 1, X.shape[1]
        self.likelihoods_0 = ((num_token_0 + smoothing) / (num_0 + 2 * smoothing))
        self.likelihoods_1 = ((num_token_1 + smoothing) / (num_1 + 2 * smoothing))

        # 각 class의 prior를 계산
        prior_probability_0 = num_0 / (num_0 + num_1)
        prior_probability_1 = num_1 / (num_0 + num_1)

        # prior의 log값 계
        self.log_prior_0 = np.log(self.likelihoods_0 / prior_probability_0)
        self.log_prior_1 = np.log(self.likelihoods_1 / prior_probability_1)

        return None

    """
    Req 3-1-5.
    predict():
    테스트 데이터에 대해서 예측 label값을 출력해주는 함수
    """

    def predict(self, X_test):
        predictions = []
        X_test=X_test.toarray()
        if (len(X_test)==1):
            predictions.append(None)
        else:
            for case in X_test:
                predictions.append(None)
        
        return predictions

    """
    Req 3-1-6.
    score():
    테스트를 데이터를 받아 예측된 데이터(predict 함수)와
    테스트 데이터의 label값을 비교하여 정확도를 계산
    """
    
    def score(self, X_test, Y_test):
        
        return None

model = Naive_Bayes_Classifier()
model.train(X, Y)
'''
# Req 3-2-1. model에 Naive_Bayes_Classifier 클래스를 사용하여 학습합니다.
model = None

# Req 3-2-2. 정확도 측정
print("Naive_Bayes_Classifier accuracy: {}".format(None))

# Logistic regression algorithm part
# 아래의 코드는 심화 과정이기에 사용하지 않는다면 주석 처리하고 실행합니다.

"""
Logistic_Regression_Classifier 알고리즘 클래스입니다.
"""

class Logistic_Regression_Classifier(object):
    
    """
    Req 3-3-1.
    sigmoid():
    인풋값의 sigmoid 함수 값을 리턴
    """
    def sigmoid(self,z):
        
        return None

    """
    Req 3-3-2.
    prediction():
    X 데이터와 beta값들을 받아서 예측 확률P(class=1)을 계산.
    X 행렬의 크기와 beta의 행렬 크기를 맞추어 계산.
    ex) sigmoid(            X           x(행렬곱)       beta_x.T    +   beta_c)       
                (데이터 수, feature 수)             (feature 수, 1)
    """

    def prediction(self, beta_x, beta_c, X):
        # 예측 확률 P(class=1)을 계산하는 식을 만든다.
    
        return None

    """
    Req 3-3-3.
    gradient_beta():
    beta값에 해당되는 gradient값을 계산하고 learning rate를 곱하여 출력.
    """
    
    def gradient_beta(self, X, error, lr):
        # beta_x를 업데이트하는 규칙을 정의한다.
        beta_x_delta = None
        # beta_c를 업데이트하는 규칙을 정의한다.
        beta_c_delta = None
    
        return beta_x_delta, beta_c_delta

    """
    Req 3-3-4.
    train():
    Logistic Regression 학습을 위한 함수.
    학습데이터를 받아서 최적의 sigmoid 함수으로 근사하는 가중치 값을 리턴.

    알고리즘 구성
    1) 가중치 값인 beta_x_i, beta_c_i 초기화
    2) Y label 데이터 reshape
    3) 가중치 업데이트 과정 (iters번 반복) 
    3-1) prediction 함수를 사용하여 error 계산
    3-2) gadient_beta 함수를 사용하여 가중치 값 업데이트
    4) 최적화 된 가중치 값들 리턴
       self.beta_x, self.beta_c
    """
    
    def train(self, X, Y):
        # Req 3-3-8. learning rate 조절
        # 학습률(learning rate)를 설정한다.(권장: 1e-3 ~ 1e-6)
        lr = 1e-2
        # 반복 횟수(iteration)를 설정한다.(자연수)
        iters = 200
        
        # beta_x, beta_c값을 업데이트 하기 위하여 beta_x_i, beta_c_i값을 초기화
        beta_x_i = None
        beta_c_i = None
    
        #행렬 계산을 위하여 Y데이터의 사이즈를 (len(Y),1)로 저장합니다.
        Y=None
    
        for i in range(iters):
            #실제 값 Y와 예측 값의 차이를 계산하여 error를 정의합니다.
            error = None
            #gredient_beta함수를 통하여 델타값들을 업데이트 합니다.
            beta_x_delta, beta_c_delta = self.gradient_beta(None)
            beta_x_i -= beta_x_delta.T
            beta_c_i -= beta_c_delta
            
        self.beta_x = beta_x_i
        self.beta_c = beta_c_i
        
        return None

    """
    Req 3-3-5.
    classify():
    확률값을 0.5 기준으로 큰 값은 1, 작은 값은 0으로 리턴
    """

    def classify(self, X_test):
        
        return None

    """
    Req 3-3-6.
    predict():
    테스트 데이터에 대해서 예측 label값을 출력해주는 함수
    """
    
    def predict(self, X_test):
        predictions = []
        X_test=X_test.toarray()
        if (len(X_test)==1):
            predictions.append(None)
        else:
            for case in X_test:
                predictions.append(None)
        
        return predictions


    """
    Req 3-3-7.
    score():
    테스트를 데이터를 받아 예측된 데이터(predict 함수)와
    테스트 데이터의 label값을 비교하여 정확도를 계산
    """
    
    def score(self, X_test, Y_test):

        return None

# Req 3-4-1. model2에 Logistic_Regression_Classifier 클래스를 사용하여 학습합니다.
model2 = None

# Req 3-4-2. 정확도 측정
print("Logistic_Regression_Classifier accuracy: {}".format(None))

'''