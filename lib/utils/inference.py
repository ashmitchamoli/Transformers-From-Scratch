import torch
import sys
import gc
from alive_progress import alive_bar as aliveBar

from lib.transformer import Transformer
from lib.config import EOS_TOKEN, PAD_TOKEN

EPSILON = 1e-20

class Inferencer:
	def __init__(self, model : Transformer) -> None:
		self.model = model
		self.vocabularyTgt = model.vocabularyTgt
	
	def getTokenIndices(self, tokens : list[str]) -> torch.Tensor:
		"""
		Returns a tensor of word indices.
		"""
		return torch.tensor([self.vocabularyTgt[token] for token in tokens], dtype=torch.long)
	
	def generateNextWordIndex(self, contextTensor : torch.Tensor = []) -> int:
		nextWordDistribution = self.model.getNextWordDistribution(contextTensor)
		return self.decodeNextWord(nextWordDistribution)

	def generateSentence(self, context : list[str] = []) -> str:
		"""
		Generates a sentence out of the given context
		"""
		currentTokenIndex = -1
		contextTensor = self.getTokenIndices(context)
		while currentTokenIndex != self.vocabularyTgt[EOS_TOKEN]:
			currentTokenIndex = self.generateNextWordIndex(contextTensor)
			
			# expand current context tensor
			contextTensor = torch.cat([contextTensor, torch.tensor([currentTokenIndex])])
			context.append(self.vocabularyTgt.inverse[currentTokenIndex])

			print(context[-1], end=" ")
			sys.stdout.flush()
		
		return " ".join(context)

	def computePerplexity(self, tokens : list[list[str]], saveToFile : bool = False, filePath : str | None = None) -> float:
		"""
		Computes the perplexity of the given list of tokens.
		"""
		perpFile = None
		if saveToFile:
			perpFile = open(filePath, "w")

		totalPerplexity = 0
		with aliveBar(len(tokens)) as bar:
			for i, sentence in enumerate(tokens):
				logSentenceProb = 0
				tokenIndices = self.getTokenIndices(sentence)

				context = torch.full((len(sentence), len(sentence)), fill_value=self.vocabularyTgt[PAD_TOKEN], dtype=torch.long)
				for i in range(1, len(tokenIndices)):
					context[i, (len(sentence) - i):] = tokenIndices[:i]
				
				probDist = self.model.getNextWordDistribution(context)
				probDist = probDist.to("cpu")

				logSentenceProb = torch.log(probDist[torch.arange(len(tokenIndices)), tokenIndices]).sum()				
				sentencePerp = torch.exp(-logSentenceProb / len(sentence))
				totalPerplexity += sentencePerp

				if saveToFile:
					perpFile.write(f"{' '.join(sentence)}\t{sentencePerp.item()}\n")

				bar.text(f"Avg. Perplexity: {(totalPerplexity.item() / (i+1)):.3f}")
				bar()

				with torch.no_grad():
					# gc.collect()
					torch.cuda.empty_cache()

		if saveToFile:
			perpFile.write(f"Avg. Perplexity: {totalPerplexity.item() / len(tokens)}\n")
			perpFile.close()

		return totalPerplexity.item() / len(tokens)

	def decodeNextWord(self, nextWordDistribution : torch.Tensor) -> int:
		"""
		Returns the index of the next word.
		"""
		return nextWordDistribution.argmax().item()

	def __call__(self, context : list[str]) -> str:
		return self.generateSentence(context)