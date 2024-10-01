import pickle
import torch

from lib.utils import TranslationDataset
from lib.transformer import Transformer

hyperparameters = {
    "dModel" : 512,
    "nHeads" : 4,
    "nEncoderLayers" : 3,
    "nDecoderLayers" : 3,
    "activation" : "gelu",
    "dropout" : 0.2,
    "dimFeedforward" : 1024,
    "normFirst" : False,
    "classifierLayers" : [1024]
}

trainingConfig = {
	"batchSize" : 4,
	"learningRate" : 0.001,
	"numEpochs" : 3
}

vocabularyEn = pickle.load(open("data/vocabularyEn.pkl", "rb"))
vocabularyFr = pickle.load(open("data/vocabularyFr.pkl", "rb"))

trainTokensEn, trainTokensFr = pickle.load(open("data/train_split.pkl", "rb"))
devTokensEn, devTokensFr = pickle.load(open("data/dev_split.pkl", "rb"))
testTokensEn, testTokensFr = pickle.load(open("data/test_split.pkl", "rb"))

trainDataset = TranslationDataset(vocabularyEn, vocabularyFr, trainTokensEn, trainTokensFr)
trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=trainingConfig["batchSize"], shuffle=True, collate_fn=trainDataset.customCollate)

valDataset = TranslationDataset(vocabularyEn, vocabularyFr, devTokensEn, devTokensFr)
valLoader = torch.utils.data.DataLoader(valDataset, batch_size=trainingConfig["batchSize"], shuffle=True, collate_fn=valDataset.customCollate)

model = Transformer(vocabularyTgt=vocabularyFr,
					vocabularySrc=vocabularyEn,
					**hyperparameters)

model.trainModel(trainLoader,
				 valLoader,
				 trainingConfig["learningRate"],
				 trainingConfig["numEpochs"])