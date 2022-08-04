import torch
from torchtext.datasets import AG_NEWS
import pandas as pd

# train_dataset, test_dataset = AG_NEWS(ngrams=3)

train_df = pd.read_csv('./.data/ag_news_csv/train.csv')
test_df = pd.read_csv('./.data/ag_news_csv/test.csv')

print(train_df.head())
print(test_df.head())

print(len(train_df))
print(len(test_df))

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer('basic_english')
train_iter = AG_NEWS(split='train')

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)


vocab = build_vocab_from_iterator(yield_tokens(train_iter), special=["<unk>"])

