
import urllib.request
import pandas as pd
from torchtext.data import TabularDataset
from torchtext import data
from torchtext.data import Iterator

def main():
    '''
    일회용
    :return:
    '''

    # urllib.request.urlretrieve("https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv", filename="IMDb_Reviews.csv")
    #
    # df = pd.read_csv('IMDb_Reviews.csv', encoding='latin1')
    # df.head()
    #
    # print(len(df))
    #
    # train_df = df[:25000]
    # test_df = df[25000:]
    #
    # train_df.to_csv("train_data.csv", index = False)
    # test_df.to_csv("test_data.csv", index=False)

    '''
    필드 정의
    필드를 통해 앞으로 어떻게 전처리를 진행할 것인지를 정해준다.
    '''
    TEXT = data.Field(sequential=True,
                      use_vocab=True,
                      tokenize=str.split,
                      lower=True,
                      batch_first=True,
                      fix_length=20)

    LABEL = data.Field(sequential=True,
                      use_vocab=True,
                      batch_first=True,
                      is_target=False)

    train_data, test_data = TabularDataset.splits(
        path = '.', train = 'train_data.csv', test = 'test_data.csv', format='csv',
        fields = [('text', TEXT), ('label', LABEL)], skip_header = True
    )

    print('train sample : {}'.format(len(train_data)))
    print('test sample : {}'.format(len(test_data)))

    print(vars(train_data[0]))

    '''
    단어 집합 만들기 :  정의한 빌드에 .build_vocab 함수 사용
    '''

    TEXT.build_vocab(train_data, min_freq=10, max_size=10000)
    # print('vocab size : {}'.format(len(TEXT.vocab)))
    # print(TEXT.vocab.stoi)

    '''
    데이터 로더 만들기
    '''
    batch_size = 5
    train_loader = Iterator(dataset = train_data, batch_size = batch_size)
    test_loader = Iterator(dataset = test_data, batch_size = batch_size)

    print("훈련 데이터 미니 배치 수 {}".format(len(train_loader)))
    print("테스트 데이터 미니 배치 수 {}".format(len(test_loader)))

    batch = next(iter(train_loader))
    # print(batch.text)


if __name__ == '__main__':
    main()


