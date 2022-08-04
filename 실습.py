# from konlpy.tag import Okt
#
# okt = Okt()
# token = okt.morphs("나는 자연어 처리를 배운다.")
# print(token)
#
# word2index = {}
#
# for voca in token:
#     if voca not in word2index.keys():
#         word2index[voca] = len(word2index)
#
# print(word2index)
#
# def one_hot_encoding(word, word2index):
#     one_hot_vector = [0]*(len(word2index))
#     index = word2index[word]
#     one_hot_vector[index] = 1
#     return one_hot_vector
#
# print(one_hot_encoding("자연어", word2index))
# print(one_hot_encoding("를", word2index))

'''-----------------------------------------------------------------
워드 임베딩(word embedding) : 단어를 밀집(dense) 표현(밀집 벡터) 으로 변환하는 방법
희소 표현 ( ex 원-핫 인코딩) -> 유사도 처리 불가능, 리소스 많이 잡아 먹음(단어의 갯수가 많아지면 많아질 수록)
밀집 표현 ( 사용자가 설정한 값으로 모든 단어의 벡터 표현의 차원을 맞춰줍니다. 0과 1이 아닌 실수값을 가지게 됩니다.)

워드 임베딩 방법론으로는 LSA, Word2Vec, FastText, Glove 등이 있음

Word2Vec에는 CBOW(Continuous Bag of  Words)와 Skip-Gram 두 가지 방식이 있음.

CBOW는 주변에 있는 단어들을 가지고, 중간에 있는 단어들을 예측하는 방법입니다.
Skip-Gram은 중간에 있는 단어로 주변 단어들을 예측하는 방법입니다.

Projection Layer에서 벡터의 평균을 구하는 부분이 차이가 있음.
Skip-Gram은 입력이 중심 단어 하나이기때문에 Projection layer에서 벡터의 평균을 구하지 않습니다.

CBOW는 주변의 있는 단어들을 입력으로 넣어 Projection layer에서 벡터의 평균을 구해 가중치를 곱한 후 입력과 같은 크기의
output vector를 만들어내게 된다. 이 벡터들을 통해 기존에 우리가 알고 있는 정답의 원 핫 벡터와 cross-entropy라는 손실함수를 통해
오차를 구한후 가중치 w와 w`를 학습해 나아가는 방식..

[Word2Vec SGNS(Skip-Gram with Negative Sampling)]
효율적으로 우리가 원하는 상관있는 단어의 임베딩만을 조절 할 수는 없을까?
상관없는 단어들을 일부만 가져올 수 있음. 즉, 전체 단어 집합보다 훨씬 작은 단어 집합을 만들어놓고
마지막 단계를 이진 분류 문제로 바꿔버리는 것.
즉, Word2Vec은 주변 단어들을 긍정(positive)으로 두고 랜덤으로 샘플링 된 단어들을 부정(negative)로 둔 다음에 이진 분류 문제를 수행합니다.
'''

'''
Glove (Global Vectors for Word Representation)
카운트 기반과 예측 기반을 모두 사용하는 방법론

윈도우 기반 동시 등장 행렬 (Window bases Co-occurrence Matrix)
행과 열을 전체 단어 집합의 단어들로 구성하고, i 단어의 윈도우 크기 내에서 k 단어가 등장한 횟수를 i행 k열에 기재한 행렬
- 윈도우 크기가 N일 때는 좌, 우에 존재하는 N개의 단어만 참고하게 됩니다.

동시 등장 확률 (Co-occurrence Probability)
동시 등장 확률 는 동시 등장 행렬로부터 특정 단어 i의 전체 등장 횟수를 카운트하고,
특정 단어 i가 등장했을 때 어떤 단어 k가 등장한 횟수를 카운트하여 계산한 조건부 확률입니다.
'''

'''
룩업 테이블 원리
'''
train_data = 'you need to know how to code'

word_set = set(train_data.split())

vocab = {word: i+2 for i, word in enumerate(word_set)}
vocab['<unk>'] = 0
vocab['<pad>'] = 1
print(vocab)

import torch
# 단어 집합의 크기만큼의 행을 가지는 테이블 생성.
embedding_table = torch.FloatTensor([
                               [ 0.0,  0.0,  0.0],
                               [ 0.0,  0.0,  0.0],
                               [ 0.2,  0.9,  0.3],
                               [ 0.1,  0.5,  0.7],
                               [ 0.2,  0.1,  0.8],
                               [ 0.4,  0.1,  0.1],
                               [ 0.1,  0.8,  0.9],
                               [ 0.6,  0.1,  0.1]])

sample = 'you need to run'.split()
idxes = []
# 각 단어를 정수로 변환
for word in sample:
  try:
    idxes.append(vocab[word])
  # 단어 집합에 없는 단어일 경우 <unk>로 대체된다.
  except KeyError:
    idxes.append(vocab['<unk>'])
idxes = torch.LongTensor(idxes)
print(idxes)
# 각 정수를 인덱스로 임베딩 테이블에서 값을 가져온다.
lookup_result = embedding_table[idxes, :]
print(lookup_result)

'''
임베딩 층 사용하기 (만들기)
'''

train_data = 'you need to know how to code'
# 중복을 제거한 단어들의 집합인 단어 집합 생성.
word_set = set(train_data.split())

# 단어 집합의 각 단어에 고유한 정수 맵핑.
vocab = {tkn: i+2 for i, tkn in enumerate(word_set)}
vocab['<unk>'] = 0
vocab['<pad>'] = 1

import torch.nn as nn
embedding_layer = nn.Embedding(num_embeddings=len(vocab), # 단어 집합의 크기
                               embedding_dim=20,  # 임베딩 할 벡터의 차원 (사용자 정의)
                               padding_idx=1) # 선택적으로 사용하는 인자. 패디을 위한 토큰의 인덱스를 알려줍니다.

# print(embedding_layer.weight)

'''
사전 훈련된 워드 임베딩 사용
'''

from torchtext import data, datasets

# Field 정의
TEXT = data.Field(sequential=True, batch_first=True, lower=True)
LABEL = data.Field(sequential=False, batch_first=True)

train_set, test_set = datasets.IMDB.splits(TEXT, LABEL)

# from gensim.models import KeyedVectors
#
#
# word2vec_model = KeyedVectors.load_word2vec_format('eng_w2v')
# print(word2vec_model['this']) # 영어 단어 'this'의 임베딩 벡터값 출력
# print(word2vec_model['self-indulgent']) # 영어 단어 'self-indulgent'의 임베딩 벡터값 출력

'''
토치테스트에서 제공하는 사전 훈련된 워드 임베딩
'''
from torchtext.vocab import GloVe

TEXT.build_vocab(train_set, vectors=GloVe(name='6B', dim=300), max_size=10000, min_freq=10)
LABEL.build_vocab(test_set)

print(TEXT.vocab.stoi)

embedding_layer = nn.Embedding.from_pretrained(TEXT.vocab.vectors, freeze=False)

