import torch
from torch import Tensor
from torch.utils.data import Dataset
from bidict import bidict

from lib.config import PAD_TOKEN

class TranslationDataset(Dataset):
	def __init__(self, vocabSrc : bidict, vocabTgt : bidict, tokensSrc : list[list[str]], tokensTgt : list[list[str]]) -> None:
		super().__init__()

		self.vocabSrc = vocabSrc
		self.vocabTgt = vocabTgt

		self.tokensSrc, self.tokensTgt = self.prepareTokens(tokensSrc, tokensTgt)

	def prepareTokens(self, tokensSrc : list[list[str]], tokensTgt : list[list[str]]) -> tuple[list[Tensor], list[Tensor]]:
		tokensSrc = [Tensor([self.vocabSrc[token] for token in sent]).long() for sent in tokensSrc]
		tokensTgt = [Tensor([self.vocabTgt[token] for token in sent]).long() for sent in tokensTgt]

		return tokensSrc, tokensTgt

	def __len__(self) -> int:
		return len(self.tokensSrc)

	def __getitem__(self, idx : int) -> tuple[Tensor, Tensor, Tensor]:
		"""
		src sequence, tgt sequence, next word predictions
		"""
		return self.tokensSrc[idx], self.tokensTgt[idx]
	
	def customCollate(self, batch : list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
		"""
		:param batch: list of (fr, en) pairs
		"""
		sent1, sent2 = zip(*batch)

		# pad
		sent1 = torch.nn.utils.rnn.pad_sequence(sent1, batch_first=True, padding_value=self.vocabSrc[PAD_TOKEN]) # batchSize, seqLen1
		sent2 = torch.nn.utils.rnn.pad_sequence(sent2, batch_first=True, padding_value=self.vocabTgt[PAD_TOKEN]) # batchSize, seqLen2

		# make the same length
		batchSize = sent1.size(0)
		if sent1.size(1) > sent2.size(1):
			sent2 = torch.cat([sent2, torch.full((batchSize, sent1.size(1) - sent2.size(1)), self.vocabTgt[PAD_TOKEN])], dim=1)
		elif sent1.size(1) < sent2.size(1):
			sent1 = torch.cat([sent1, torch.full((batchSize, sent2.size(1) - sent1.size(1)), self.vocabSrc[PAD_TOKEN])], dim=1)

		# print("From Collate:", sent1.shape, sent2.shape)

		return sent1, sent2 # (batchSize, seqLen), (batchSize, seqLen)