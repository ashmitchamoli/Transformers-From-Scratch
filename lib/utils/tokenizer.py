import nltk
from bidict import bidict
from itertools import chain
from typing import Literal

from lib.config import PAD_TOKEN, UNK_TOKEN, EOS_TOKEN, SOS_TOKEN

class Tokenizer:
	def __init__(self) -> None:
		self.vocabulary = bidict()
		self.vocabSize = 0
		self.numTokens = 0
	
	def readText(self, filepath) -> str:
		with open(filepath, "r", encoding="utf-8") as f:
			return f.read()
	
	def updateVocab(self, word) -> None:
		if word not in self.vocabulary:
			self.vocabulary[word] = self.vocabSize
			self.vocabSize += 1

	def getTokens(self, language : Literal["english", "french"], putEos : bool = True, filepath : str = "") -> tuple[list[list[str]], bidict]:
		"""
			:param replaceUnk: if true, replaces all infrequent words with UNK_TOKEN

		returns a list of tokenized sentences.
		"""
		# self.vocabulary = bidict()
		text = self.readText(filepath)

		sentences = nltk.sent_tokenize(text, language=language)
		tokens = [([SOS_TOKEN] if putEos else []) + 
				  nltk.word_tokenize(sentence, language=language) + 
				  ([EOS_TOKEN] if putEos else []) for sentence in sentences]

		uniqueWords = set(list(chain(*tokens)))
		for word in uniqueWords:
			self.updateVocab(word)
		self.updateVocab(PAD_TOKEN)
		self.updateVocab(UNK_TOKEN)

		return tokens, self.vocabulary
