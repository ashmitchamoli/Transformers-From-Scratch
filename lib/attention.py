from torch import Tensor, matmul
from torch.nn import Module, Linear, Dropout, Softmax
from torch.nn import MultiheadAttention

class MultiHeadAttention(Module):
	def __init__(self, dModel : int = 512, nHeads : int = 8):
		super().__init__()

		try:
			assert dModel // nHeads * nHeads == dModel
		except AssertionError:
			raise ValueError("dModel must be divisible by nHeads")
		
		self.dModel = dModel
		self.nHeads = nHeads
		self.headSize = dModel // nHeads

		self.Wq = Linear(dModel, dModel)
		self.Wk = Linear(dModel, dModel)
		self.Wv = Linear(dModel, dModel)
		self.attentionSoftmax = Softmax(dim=-1)

	def forward(self, q : Tensor, k : Tensor, v : Tensor, mask : Tensor = None) -> Tensor:
		"""
		:param q: (batchSize, seqLen, dModel)
		:param k: (batchSize, seqLen, dModel)
		:param v: (batchSize, seqLen, dModel)
		:param mask: (seqLen, seqLen)
		:param normFirst: bool

		:return: (batchSize, seqLen, dModel)
		"""
		batchSize, seqLen, _ = q.shape

		q = self.Wq(q) # (batchSize, seqLen, dModel)
		k = self.Wk(k) # (batchSize, seqLen, dModel)
		v = self.Wv(v) # (batchSize, seqLen, dModel)

		# divide into nHeads
		q = q.view(batchSize, self.nHeads, seqLen, self.headSize)
		k = k.view(batchSize, self.nHeads, seqLen, self.headSize)
		v = v.view(batchSize, self.nHeads, seqLen, self.headSize)

		attentionWeights = matmul(q, k.transpose(2, 3)) / self.headSize**0.5 # (batchSize, nHeads, seqLen, seqLen)
		if mask is not None:
			attentionWeights = attentionWeights + mask
		attentionWeights = self.attentionSoftmax(attentionWeights) # (batchSize, nHeads, seqLen, seqLen)

		finalOut = attentionWeights * v # (batchSize, nHeads, seqLen, headSize)
		finalOut = finalOut.view(batchSize, seqLen, self.dModel)

		return finalOut