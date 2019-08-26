# 분류 알고리즘을 활용한 자연어 처리

## 1  프로젝트 개요

> Sub PJT 2 에서는 문장 데이터를 training 및 test 데이터로 사용하기에 자연어 처리 과정을 통하여 머신 러닝 알고리즘에 사용할 수 있게 변환합니다. 
>
> 영화 평점 데이터의 label은 긍정적인 평점인 1과 부정적인 평점 0 값으로 이루어져 있기에 지도학습 중에서 분류(classification) 알고리즘인 Logistic regression 과 Naive bayes classifier을 사용하여 학습합니다. 



### 1. 머신 러닝 파트

> 머신 러닝 파트에서는 영화 댓글에 따른 평점 분석기를 구현합니다. 머신 러닝의 구조는 5가지 파트로 분류할 수 있습니다

1. 데이터를 읽기

   * ratings_train.txt, ratings_test.txt 파일을 읽고 트레이닝 데이터와 테스트 데이터로 저장합니다.
   * ratings_train.txt파일은 영화 댓글과 그에 따른 label 값인 0 또는 1이 저장되어 있습니다.
   * 첫 번째 열을 제외한 document, laber 열을 읽습니다.

2. 문장 데이터 자연어 처리하기

   > 전 처리 과정 : tokenizing -> word_indices -> one hot embedding -> sparse matrix

   * 문장 데이터를 처리하기 위해서 자연어 처리 과정을 거쳐 컴퓨터가 처리할 수 있는 데이터로 변환합니다.

   * 문자을 형태소 별로 나누기 위하여 KoNLPy를 사용하여 tokenizing을 합니다.

   * One-hot 임베딩을 구현하기 위해서 모든 token들의 정보가 필요하고 이는 word_indices라는 사전 데이터로 저장합니다.

     > **One-hot 임베딩** : 학습에 사용된 모든 token의 중류들 중에 트레이닝 문장에서 사용되는 token에 1을 붙이고 나머지를 0으로 채우는 방식

   * One-hot 임베딩 과정으로 만들어진 행렬의 크기가 매우 크기 때문에 머신 러닝 모델을 학습할 시 복잡도 문제를 덜고자 sparse matrix를 사용합니다.

     > **Sparse matrix** : 유의미한 데이터보다 무의미한 데이터 즉 0값이 많은 비중을 차지하는 행렬의 경우 그 위의 값을 갖는 데이터의 인덱스 정보만들 기억하여 메모리를 효율적으로 사용하는 기법

3. 알고리즘을 불러와 학습

   * 분류 기법 중 선형으로 데이터를 근사화하는 기법으로 logistic regression과 naive bayes classifier 모델을 사용하고 데이터를 학습합니다.

4. 학습된 결과물로 테스트 데이터를 통하여 정확도 측정   [참고](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)

   * 학습된 결과물로 clf, clf2, word_indices를 얻을 수 있으며, 각각 Logistic regression 모델, Naive bayes classifier 모델, token 사전 데이터를 의미합니다.
   * 이후, 테스트 데이터를 사용하여 정확도를 확인합니다. Scikit-learn에서 multiclass classification의 정확도를 구하기 위해서 제공하는 accuacy_score를 사용합니다.

5. 학습된 결과물을 저장

   * 학습된 Logistic regression과 Naive bayes classifier모델의 class 정보를 파이썬의 pickle 기능을 통하여 model.clf 에 저장합니다. Flask 서버 코드에서는 이 정보를 불러와 사용하여 학습의 반복을 피할 수 있습니다.



### 2. Slack app 구현 파트

> database를 사용하여 새로운 데이터들을 저장하고 이를 활용하여 새로운 학습 데이터로서 활용할 수 있습니다.
>
> 이번 프로젝트에서는 간단하게 slack에서 입력하는 영화 댓글의 문장을 app.db에 저장하는 것을 목표로 합니다.
>
> 이후, 이를 활용하여 입력하는 댓글의 평점 또한 받게 하여 수 많은 사람들이 이 영화 평점 분석기를 이용했을 시 얻어지는 database 정보를 가지고 새로운 트레이닝 셋을 업데이트 할 수 있습니다.



### 3. Naive bayes classifier algorithm 구현 파트

> 이 과정은 Naive bayes classifier algorithm의 심도있는 이해를 위하여 그 동작 과정을 구현합니다.

1. `log_likelihoods_naivebayes() ` : Feature 데이터를 받아 label(class)값에 해당되는 likelihood 값들을 naive방식으로 구하고 그 값의 로그 값을 리턴
2. `class_posteriors()` : Feature 데이터를 받아 label(class) 값에 해당되는 posterior 값들을 구하고 그 값으 로그 값을 리턴
3. `classify()` : Feature 데이터에 해당되는 posterior 값들(class 개수)을 불러와 비교하여 더 높은 확률을 갖는 class를 리턴
4. `train()` : 트레이닝 데이터를 받아 학습합니다. 학습 후, 각 class에 해당하는 prior 값과 likelihood 값을 업데이트
5. `pridict()` : 예측하고 싶은 문장의 전 처리된 행렬을 받아 예측된 class를 리턴
6. `score()` : 테스트 데이터를 받아 예측된 데이터(predict 함수)와 테스트 데이터의 label값을 비교하여 정확도를 계산



### 4. Logistic regression algorithm 구현 파트

> 이 과정을 심화 과정으로 Logistic regression algorithm의 심도있는 이해를 위하여 그 동작 과정을 구현합니다. 알고리즘은 Logistic_regression_Classifier 클래스를 구현하여 작동됩니다.

1. `sigmoid()` : input 값의 sigmoid함수 값을 리턴
2. `prediction()` : 가중치 값인 beta 값들을 받아서 예측 값 P(class=1|train data)을 계산. 예측 값 계산은 데이터와 가중치 간의 선형합의 logistic function 값으로 얻을 수 있음
3. `gradient_beta()` : 가중치 값인 beta 값에 해당되는 gradient 값을 계산하고 learning rate를 곱하여 출력. Gradient 값은 데이터 포인터에 대한 기울기를 의미하고 손실함수에 대한 가중치 값의 미분 형태로 얻을 수 있음
4. `train()` : 트레이닝 데이터를 받아 학습. 학습 후 sigmoid 함수로 근사하는 최적의 가중치 값을 업데이트
5. `predict()` : 예측하고 싶은 문장의 전 처리된 행렬을 받아 예측된 class를 리턴
6. `score()` : 테스트 데이터를 받아 예측된 데이터(predict 함수)와 테스트 데이터의 label 값을 비교하여 정확도를 계산



## 2  프로젝트 목표

1. Classification을 사용하여 영화 댓글 평점 분석 챗봇 구현하기
   1. 영화 댓글, 평점 데이터를 읽고 트레이닝 데이터와 테스트 데이터로 저장하기
   2. 한글 문장 데이터 tokenizing하기
   3. token 정보가 저장된 사전을 만들고 one-hot 임베딩하기
   4. 임베딩 된 데이터를 sparse matrix로 저장하기
   5. scikit-learn에서 logistic regression과 Naive bayes classifier 모델을 가져와 학습하기
   6. 정확도 평가를 위한 accuracy 값 구하기
   7. 테스트 데이터의 영화 댓글에 따른 긍정, 부정 리뷰 분류하기
   8. pickle을 사용하여 학습된 classification 모델이 저장된 model.clf 파일 저장하기
   9. Python slack client를 사용하여 flask 서버 app.py 구현하기
   10. Slack app에서 영화 댓글을 입력하고 긍정 or 부정 리뷰 출력하기
   11. SQLite를 사용해서 slack에 입력받을 데이터를 저장할 app.db 만들기
   12. app.db에 새로운 데이터 업데이트 하기

2. Naive bayes classifier algorithm 구현(심화 과정)
   1. Naive_Bayes_Classifier() 클래스 구현하기
   2. train() 함수를 사용하여 트레이닝 데이터 학습하기
   3. predict() 함수를 사용하여 테스트 데이터의 영화 댓글에 따른 긍정, 부정 리뷰 분류하기
   4. score() 함수를 사용하여 테스트 데이터 정확도 측정하기
3. Logistic regression algorithm 구현(심화 과정)
   1. Logistic_Regression_Classifier() 클래스 구현하기
   2. train() 함수를 사용하여 트레이닝 데이터 학습하기
   3. predict() 함수를 사용하여 테스트 데이터의 영화 댓글에 따른 긍정, 부정 리뷰 분류하기
   4. score() 함수를 사용하여 테스트 데이터 정확도 측정하기



## 3  기능 명세

### 1. 머신 러닝 파트

#### Req. 1-1 영화 댓글, 평점 데이터 전 처리

1. read_data() 함수를 작성하여 ratings_train.txt 파일로 저장된 트레이닝 데이터와 ratings_test.txt 파일로 저장된 트레이닝 데이터를 읽기

   > train_data : 트레이닝용 문장 및 label 데이터
   >
   > test_data : 테스트용 문장 및 label 데이터

2. 트레이닝, 테스트의 한글 문장 데이터를 KoNLPy를 사용한 toeknize() 함수를 이용하여 toeknization 구현

   > input : 텍스트 데이터
   >
   > output : ['token1/tagging', 'token2/tagging',  ...] 토큰에 해당되는 품사를 태깅

3. 단어 사전 정보가 저장된 word_inices 행렬을 생성

   > word_indices : token 사전 딕셔너리(token1:1, token2:2, ... )

4. 전 처리된 문장 데이터 행렬의 계산 복잡도를 완화할 Sparse matrix 구현

   > X, X_test : 각각 트레이닝, 테스트용 X 데이터에 해당되는 sparse 매트릭스로 초기화, 크기는 (X 데이터 수, 단어 사전의 토큰 수)
   >
   > Y, Y_test : 각각 트레이닝, 테스트용 Y 데이터에 해당되는 numpy array로 초기화, 크기는 (Y 데이터 수)

5. word_indices 행렬을 사용하여 토큰화 된 트레이닝, 테스트 데이터를 One-hot 임베딩 적용

   > X, X_test에 One-hot 임베딩 값 채우기
   >
   > Y, Y_test에 label 값 채우기

#### Req. 1-2  Classification 모델 학습

1. Scikit-learn을 활용하여 Naive bayes classifier 모델로 트레이닝 데이터 학습

   > clf에 MultinomialNB 객체 저장
   >
   > Scikit-learn에서 제공하는 함수를 사용하여 학습 데이터를 학습

2. Scikit-learn을 활용하여 Logistic regression 모델로 트레이닝 데이터 학습

   > clf2에 LogisticRegression 객체 저장
   >
   > Scikit-learn에서 제공하는 함수를 사용하여 학습 데이터를 학습

#### Req. 1-3  테스트 데이터 정확도 계산

1. 특정 문장 데이터와 Naive bayes classifier, Logistic regression 모델을 통하여 예측된 결과 값을 출력

   > 테스트 데이터 중 특정 문장과 예측된 분류 값 측정

2. 전 처리된 테스트 데이터를 통하여 Naive bayes classifier, Logistic regression 모델의 정확도 계산

   > 예측된 분류 값과 실제 label 사이의 오차의 평균을 계산(관련 함수를 찾아 사용)

#### Req 1-4  학습된 모델 저장

1. 학습된 모델(model)을 저장하기 위하여 pickle 사용
2. 학습된 모델의 class가 저장된 model.clf 파일 생성(다양한 모델을 저장 가능, 챗봇 구현에 필요한 학습 정보에 맞게 자유롭게 저장)



### 2. 웹 구현 및 데이터베이스 파트

#### Req. 2-1 웹에서 주고받는 데이터를 저장할 데이터 베이스 파일 초기화

1. SQLite를 이용하여 새롭게 입력 받을 데이터를 저장할 DB파일 생성
2. db_init.py를 사용하여 텍스트 데이터를 저장할 app.db 파일 생성(추가적으로 입력 유저 정보 및 평점까지 받을 수 DB 파일 설정 가능)

#### Req. 2-2 웹 서버 구현을 위한 flask 서버 구현

1. pickle로 저장된 model.clf 파일 불러오기

   > pickle_obj : pickle 파일 객체를 받는 변수
   >
   > word_indices : 단어 사전
   >
   > clf or clf2 : 학습 모델 객체
   >
   > slack app 기능에 맞게 예측에 필요한 추가 데이터 사용 가능

2. 입력 받은 문장 데이터를 토큰화 및 one-hot 임베딩하는 전 처리 함수 preprocess() 구현

   > input : 영화평 문장
   >
   > output : 전 처리된 sparse matrix

3. 전 처리된 문장 데이터를 긍정 혹은 부정으로 분류하는 함수 classify() 구현

   > input : 전 처리된 sparse matrix
   >
   > output : 부정적 리뷰 or 긍정적 리뷰(다양한 방식으로 분류 값 사용 가능)

4. app.db를 연동하여 웹에서 주고받는 데이터를 DB로 저장

#### Req. 2-3 Slack app에서 영화평 문장 데이터를 입력하고 예상 분류 값 출력

1. Slack app 등록된 워크스페이스에서 app을 호출하여 문장 데이터를 입력(자유로운 방식으로 문장 데이터를 받는 부분 구현)

2. Flask에서 학습된 모델 기반으로 분류된 데이터 출력(자유로운 방식으로 출력 방식 구현), 입력된 데이터로 DB 업데이트

   > 예시) 영화 댓글 관련 텍스트 입력 -> 부정 혹은 긍정에 대한 출력 받기

3. 1가지 이상의 새로운 기능을 추가한 영화 평점 분석기 챗봇 구현

   > 예시)
   >
   > 1. 출력된 분류 값이 맞지 않는 경우를 고려하여 정확한 분류 값을 새롭게 입력할 수 있는 기능
   > 2. 입력 데이터 외에 챗봇의 분류 값 데이터 또한 저장하여 새로운 데이터 셋을 구하는 기능
   > 3. 웹 크롤링을 바탕으로 새로운 데이터 셋을 만들어 다양한 학습 데이터 셋을 선택할 수 있는 기능
   > 4. 현재 구현된 알고리즘 외에 다양한 머신 러닝 알고리즘을 선택하여 결과를 비교할 수 있는 기능



### 3. 알고리즘 파트

#### Req. 3-1 Naive bayes classifier algorithm 구현

1. Likelihood 확률의 로그 값을 출력하는 log_likelihood_naivebayes() 함수 구현

   > input : 
   >
   > ​	feature_vector : 단어 사전 크기를 갖는 벡터
   >
   > ​	one-hot 임베딩을 사용했기에 0 or 1 성분 값을 갖음
   >
   > ​	Class : label 값인 0 or 1
   >
   > output : 
   >
   > ​	Class에 해당되는 likelihood 확률의 로그 값 = log(P(feature_vector | class))

2. Posterior 확률의 로그 값을 출력하는 class_posteriors() 함수 구현

   > input : feature_vector
   >
   > output : 
   >
   > ​	Class 0에 해당하는 posterior 확률의 로그 값 = log(P(class=0| feature_vector ))
   >
   > ​	Class 1에 해당하는 posterior 확률의 로그 값 = log(P(class=1| feature_vector ))

3. 입력 확률들을 바탕으로 Class(label)를 분류 해주는 classify() 함수 구현

   > input : feature_vector
   >
   > output : 0 or 1

4. 학습을 위한 train() 함수 구현

   > input : 학습용 X 데이터, 학습용 label 데이터

5. 전 처리된 문장 데이터의 예측 분류 값을 출력하는 predict() 함수 구현

   > input : 학습용 X 데이터
   >
   > output : 예측된 label 값 = (len(X_test), 1) 크기를 갖는 numpy array 행렬

6. 모델의 정확도를 계산하는 score() 함수 구현

   > input : 테스트용  X 데이터, 테스트용 label 데이터
   >
   > output : 정확히 예측된 데이터 개수의 합 / 총 테스트 데이터 수

7. Smoothing 기법을 적용하여 정확도 비교

#### Req. 3-2 Naive bayes classifier 학습 및 정확도 계산

1. train() 함수를 활용하여 트레이닝 데이터를 학습

   > self.log_prior_0, self.log_prior_1 값 업데이트
   >
   > self.likelihoods_0, self.likelihoods_1값 업데이트

2. score() 함수를 사용하여 최종적으로 트레이닝 된 데이터와 실제 데이터의 라벨을 비교하여 정확도 계산 및 출력

#### Req. 3-3 Logistic regression algorithm 구현

1. 인풋 값의 sigmoid 함수 값을 출력하는 sigmoid() 함수 구현

   > input : 실수형 벡터 혹은 행렬
   >
   > output : 인풋 값 각 성분의 sigmoid 계산 값이 저장된 사이즈와 같은 벡터 혹은 행렬

2. 데이터와 가중치 값을 받아서 예측 값 P(class = 1|train data)을 계산하는 prediction() 함수 구현

   > input : 학습용 X 데이터
   >
   > output : 예측된 확률 값 = (len(Y), 1) 크기를 갖는 numpy array 행렬

3. 가중치 값을 업데이트 하는 gradient_beta() 함수 구현

   > input : 학습용 X 데이터, error
   >
   > output : 가중치 값에 해당되는 gradient * learning

4. 학습을 위한 train() 함수 구현

   > input : 학습용 X 데이터, 학습용 label 데이터

5. 확률 값을 0.5 기준으로 0 또는 1 값으로 분류하는 classify() 함수 구현

   > input : 하나의 테스트용 X 데이터
   >
   > output : 0 or 1

6. 전 처리된 문장 데이터의 예측 분류 값을 출력하는 predict() 함수 구현

   > input : 테스트용 X 데이터
   >
   > output : 예측된 label 값 = (len(X_test), 1) 크기를 갖는 numpy array 행렬

7. 모델의 정확도를 계산하는 score() 함수 구현

   > input : 테스트용 X 데이터, 테스트용 label 데이터
   >
   > output : 정확히 예측된 데이터 개수의 합 / 총 테스트 데이터 수

8. 정확도 값이 0.7 이상이 나오도록 초기값 및 학습률을 조절

#### Req. 3-4 Logistic regression algorithm 학습 및 정확도 계산

1. train() 함수를 활용하여 트레이닝 데이터를 학습

   > 가중치 값인 self.beta_x, self.beta_c 업데이트

2. score() 함수를 사용하여 최종적으로 트레이닝 된 데이터와 실제 데이터의 라벨을 비교하여 정확도 계산 및 출력





