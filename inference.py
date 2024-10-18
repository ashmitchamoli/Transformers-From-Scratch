import pickle
import torch
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from alive_progress import alive_bar as aliveBar

from lib.utils import TranslationDataset
from lib.transformer import Transformer
from lib.config import PAD_TOKEN

hyperparameters = {
    "dModel" : 64,
    "nHeads" : 1,
    "nEncoderLayers" : 1,
    "nDecoderLayers" : 1,
    "activation" : "relu",
    "dropout" : 0.2,
    "dimFeedforward" : 256,
    "normFirst" : False,
    "classifierLayers" : []
}

vocabularyEn = pickle.load(open("data/vocabularyEn.pkl", "rb"))
vocabularyFr = pickle.load(open("data/vocabularyFr.pkl", "rb"))

testTokensEn, testTokensFr = pickle.load(open("data/test_split.pkl", "rb"))
# testDataset = TranslationDataset(testTokensEn, testTokensFr)
# testLoader = torch.utils.data.DataLoader(testDataset, batch_size=trainingConfig["batchSize"], shuffle=False)

model = Transformer(vocabularyTgt=vocabularyFr,
					vocabularySrc=vocabularyEn,
					**hyperparameters)

model.loadModelWeights()
print("Model loaded successfully.")

def calculateBleu(model : Transformer) -> list[float]:
	blueScores = []
	totalCorrect = 0
	total = 0
	with torch.no_grad():
		model.to(model.device)
		with aliveBar(len(testTokensEn)) as bar:
			for i, (src, tgt) in enumerate(zip(testTokensEn, testTokensFr)):
				src = [vocabularyEn[x] for x in src]
				tgt = [vocabularyFr[x] for x in tgt]
				if len(src) < len(tgt):
					src += [vocabularyEn[PAD_TOKEN]] * (len(tgt) - len(src))
				if len(tgt) < len(src):
					tgt += [vocabularyFr[PAD_TOKEN]] * (len(src) - len(tgt))

				src = torch.tensor([src], dtype=torch.long).to(model.device) # (1, seqLen)
				tgt = torch.tensor([tgt], dtype=torch.long).to(model.device) # (1, seqLen)

				output = model(src[:, :-1], tgt[:, :-1]).argmax(dim=-1).view(-1) # (seqLen-1)
				y = tgt[:, 1:].reshape(-1) # (seqLen-1)
				padIndices = y == vocabularyFr[PAD_TOKEN]
				y = y[~padIndices]
				output = output[~padIndices]

				score = sentence_bleu([y.tolist()], output.tolist())
				blueScores.append(score)

				totalCorrect = totalCorrect + (output == y).sum().item()
				total = total + len(y)

				bar()
	
	return blueScores + [sum(blueScores) / len(blueScores)], totalCorrect / total

bleuSores, accuracy = calculateBleu(model)

print(f"Bleu Score: {bleuSores[-1] :.3f}\nAccuracy: {accuracy:.3f}")

# with open("data/ted-talks-corpus/test.fr", "r") as f:
# 	lines = f.readlines()

# bleuScores, _ = calculateBleu(model)

# with open("testbleu.txt", "w") as f:
# 	assert len(lines) == len(bleuScores)-1
# 	for i in range(len(lines)):
# 		f.write(f"{lines[i][:-1]} {bleuScores[i]}\n")