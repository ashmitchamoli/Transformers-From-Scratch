import nltk
from torch import Tensor
from torch.utils.data import Dataset
from typing import Literal

class TedTalksDataset(Dataset):
	def __init__(self, split : Literal["train", "dev", "test"]) -> None:
		super().__init__()

		with open(f"data/ted-talks-corpus/{split}.fr", "r") as f:
			self.fr = f.read().split("\n")
		with open(f"data/ted-talks-corpus/{split}.en", "r") as f:
			self.en = f.read().split("\n")
		try:
			assert len(self.fr) == len(self.en)
		except AssertionError:
			raise ValueError("Unequal number of lines in the dataset")

	def __len__(self) -> int:
		return len(self.fr)

	def __getitem__(self, idx : int) -> tuple[Tensor, Tensor]:
		return self.fr[idx], self.en[idx]
	
	def customCollate(self, batch : list[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
		"""
		:param batch: list of (fr, en) pairs
		"""
		frs, ens = zip(*batch)