import copy
from torch import Tensor
from torch.nn import Module, LayerNorm, Dropout, Linear, ReLU, GELU, Tanh, Sequential, ModuleList
from typing import Optional, Literal

from lib.transformer.attention import MultiHeadAttention

class Encoder(Module):
	def __init__(self, encoderLayer : "EncoderLayer", nLayers : int = 6) -> None:
		super().__init__()

		self.nLayers = nLayers
		self.layers = ModuleList([copy.deepcopy(encoderLayer) for _ in range(nLayers)])

	def forward(self, x : Tensor, srcMask : Optional[Tensor] = None) -> Tensor:
		for layer in self.layers:
			x = layer(x, srcMask)
		# x = self.layers(x, srcMask)
		return x

class EncoderLayer(Module):
	def __init__(self, dModel : int = 512, 
			  	 nHeads : int = 8, 
				 activation : Literal["relu", "gelu", "tanh"] = "relu",
				 dropout : float = 0.1, 
				 dimFeedforward : int = 2048,
				 normFirst : bool = False) -> None:
		super().__init__()

		self.dModel = dModel
		self.nHeads = nHeads
		self.activation = activation
		self.dropout = dropout
		self.dimFeedforward = dimFeedforward
		self.normFirst = normFirst
		if activation == "relu":
			self.activation = ReLU()
		elif activation == "gelu":
			self.activation = GELU()
		elif activation == "tanh":
			self.activation = Tanh()


		self.selfAttention = MultiHeadAttention(dModel, nHeads)
		self.layerNorm1 = LayerNorm(dModel)
		self.layerNorm2 = LayerNorm(dModel)
		self.dropoutLayer = Dropout(dropout)
		self.linear1 = Linear(dModel, dimFeedforward)
		self.linear2 = Linear(dimFeedforward, dModel)
	
	def forward(self, x : Tensor, srcMask : Optional[Tensor] = None) -> Tensor:
		"""
		:param x: (batchSize, seqLen, dModel)
		:param srcMask: (seqLen, seqLen)
		:return: (batchSize, seqLen, dModel)
		"""
		if self.normFirst:
			x = x + self.saBlock(self.layerNorm1(x), srcMask)
			x = x + self.ffBlock(self.layerNorm2(x))
		else:
			x = self.layerNorm1(x + self.saBlock(x, srcMask))
			x = self.layerNorm2(x + self.ffBlock(x))
		
		return x
	
	def saBlock(self, x : Tensor, srcMask : Optional[Tensor] = None) -> Tensor:
		return self.dropoutLayer(self.selfAttention(x, x, x, srcMask))
	
	def ffBlock(self, x : Tensor) -> Tensor:
		return self.dropoutLayer(self.linear2(self.dropoutLayer(self.activation(self.linear1(x)))))
