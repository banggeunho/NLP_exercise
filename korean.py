
import urllib.request
import pandas as pd
from torchtext.data import Iterator

def main():
    urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt",
                               filename="ratings_train.txt")
    urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt",
                               filename="ratings_test.txt")

    train_df = pd.read_table('ratings_train.txt')
    test_df = pd.read_table('ratings_test.txt')

    print(train_df[:5])

    from torchtext import data
    from konlpy.tag import Mecab

    tokenizer = Mecab()

    ID = data.Field(sequential= False,
                    use_vocab= False)

    TEXT = data.Field(sequential=True,
                      use_vocab=True,
                      tokenize= tokenizer.morphs,
                      lower = True,
                      batch_first= True,
                      fix_length= 20)

    LABEL = data.Field(sequential= False,
                       use_vocab= False,
                       is_target = True)

    from torchtext.data import TabularDataset

    train_data, test_data = TabularDataset.splits(
        path='.', train = 'ratings_train.txt', test = 'ratings_test.txt', format='tsv',
        fields=[('id', ID), ('text', TEXT), ('label', LABEL)], skip_header = True
    )

    print(len(train_data))
    print(len(test_data))
    print(vars(train_data[0]))

    # 단어 집합 만들기

    TEXT.build_vocab(train_data, min_freq = 10, max_size = 10000)
    print(len(TEXT.vocab))
    print(TEXT.vocab.stoi)

    from torchtext.data import Iterator
    batch_size = 5
    train_loader = Iterator(dataset=train_data, batch_size = batch_size)
    test_loader = Iterator(dataset=test_data, batch_size = batch_size)




if __name__ == '__main__':
    main()


