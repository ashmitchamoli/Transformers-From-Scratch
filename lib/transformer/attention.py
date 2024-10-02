from torch import Tensor, matmul
from torch.nn import Module, Linear, Softmax

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

		# print(q.shape, self.Wq.weight.shape)
		# print(q.device, self.Wq.weight.device)

		q = self.Wq(q) # (batchSize, seqLen, dModel)
		k = self.Wk(k) # (batchSize, seqLen, dModel)
		v = self.Wv(v) # (batchSize, seqLen, dModel)

		# divide into nHeads
		q = q.view(batchSize, seqLen, self.nHeads, self.headSize).transpose(1, 2)
		k = k.view(batchSize, seqLen, self.nHeads, self.headSize).transpose(1, 2)
		v = v.view(batchSize, seqLen, self.nHeads, self.headSize).transpose(1, 2)

		x = matmul(q, k.transpose(-2, -1)) / self.headSize**0.5 # (batchSize, nHeads, seqLen, seqLen)
		if mask is not None:
			x = x + mask
		x = self.attentionSoftmax(x) # (batchSize, nHeads, seqLen, seqLen)

		x = matmul(x, v) # (batchSize, nHeads, seqLen, headSize)
		x = x.transpose(1, 2).contiguous().view(batchSize, seqLen, self.dModel)

		return x