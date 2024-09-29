import copy
import torch
from torch import Tensor
from torch.nn import Module, LayerNorm, Dropout, Linear, ReLU, GELU, Tanh, TransformerDecoder
from typing import Optional, Literal

from .attention import MultiHeadAttention

class Decoder(Module):
	def __init__(self, decoderLayer : "DecoderLayer", nLayers : int = 6) -> None:
		super().__init__()

		self.nLayers = nLayers
		self.layers = [copy.deepcopy(decoderLayer) for _ in range(nLayers)]

	def forward(self, x : Tensor, q : Tensor, k : Tensor, causalMask : Tensor) -> Tensor:
		for layer in self.layers:
			x = layer(x, q, k, causalMask)
		return x

class DecoderLayer(Module):
	def __init__(self, dModel : int = 512, 
			  	 nHeads : int = 8, 
				 activation : Literal["relu", "gelu", "tanh"] = "relu",
				 dropout : float = 0.1, 
				 dimFeedforward : int = 2048,
				 normFirst : bool = False) -> None:
		super().__init__()

		self.dModel = dModel
		self.nHeads = nHeads
		self.dropout = dropout
		self.dimFeedforward = dimFeedforward
		self.normFirst = normFirst
		self.activation = activation
		if activation == "relu":
			self.activation = ReLU()
		elif activation == "gelu":
			self.activation = GELU()
		elif activation == "tanh":
			self.activation = Tanh()

		self.maskedAttention = MultiHeadAttention(dModel, nHeads)
		self.crossAttention = MultiHeadAttention(dModel, nHeads)
		self.layerNorm1 = LayerNorm(dModel)
		self.layerNorm2 = LayerNorm(dModel)
		self.layerNorm3 = LayerNorm(dModel)
		self.linear1 = Linear(dModel, dimFeedforward)
		self.linear2 = Linear(dimFeedforward, dModel)
		self.dropoutLayer = Dropout(dropout)

	def forward(self, x : Tensor, q : Tensor, k : Tensor, causalMask : Tensor) -> Tensor:
		"""
		:param x: (batchSize, seqLen, dModel)
		:param q: (batchSize, seqLen, dModel)
		:param k: (batchSize, seqLen, dModel)
		:param causalMask: (seqLen, seqLen), mask for masked attention
		:return: (batchSize, seqLen, dModel)
		"""
		if self.normFirst:
			x = x + self.saBlock(self.layerNorm1(x), causalMask) # masked attention
			x = x + self.caBlock(self.layerNorm2(x), q, k) # cross attention
			x = x + self.ffBlock(self.layerNorm2(x)) # feed forward
		else:
			x = self.layerNorm1(x + self.saBlock(x))
			x = self.layerNorm2(x + self.caBlock(x, q, k))
			x = self.layerNorm2(x + self.ffBlock(x))
		
		return x
	
	def saBlock(self, x : Tensor, srcMask : Optional[Tensor] = None) -> Tensor:
		return self.dropoutLayer(self.selfAttention(x, x, x, srcMask))
	
	def caBlock(self, x : Tensor, q : Tensor, k : Tensor) -> Tensor:
		return self.dropoutLayer(self.crossAttention(q, k, x))

	def ffBlock(self, x : Tensor) -> Tensor:
		return self.dropoutLayer(self.linear2(self.dropoutLayer(self.activation(self.linear1(x)))))