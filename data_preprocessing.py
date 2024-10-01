import pickle
from lib.utils import Tokenizer
from bidict import bidict

tokenizer = Tokenizer()

trainTokensEn, vocabularyEn = tokenizer.getTokens(filepath="data/ted-talks-corpus/train.en")
testTokensEn, _ = tokenizer.getTokens(filepath="data/ted-talks-corpus/test.en")
devTokensEn, _ = tokenizer.getTokens(filepath="data/ted-talks-corpus/dev.en")

tokenizer.vocabulary = bidict()
trainTokensFr, vocabularyFr = tokenizer.getTokens(filepath="data/ted-talks-corpus/train.fr")
testTokensFr, _ = tokenizer.getTokens(filepath="data/ted-talks-corpus/test.fr")
devTokensFr, _ = tokenizer.getTokens(filepath="data/ted-talks-corpus/dev.fr")

train_split = (trainTokensEn, trainTokensFr)
dev_split = (devTokensEn, devTokensFr)
test_split = (testTokensEn, testTokensFr)

with open("data/train_split.pkl", "wb") as f:
	pickle.dump(train_split, f)

with open("data/dev_split.pkl", "wb") as f:
	pickle.dump(dev_split, f)

with open("data/test_split.pkl", "wb") as f:
	pickle.dump(test_split, f)						

with open("data/vocabularyEn.pkl", "wb") as f:
	pickle.dump(vocabularyEn, f)

with open("data/vocabularyFr.pkl", "wb") as f:
	pickle.dump(vocabularyFr, f)