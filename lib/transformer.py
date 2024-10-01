import torch
from torch import Tensor, inf as infinity
from torch.nn import Module, Transformer
from typing import Literal

from .encoder import Encoder, EncoderLayer
from .decoder import Decoder, DecoderLayer

class Transformer(Module):
	def __init__(self,
				 dModel : int = 512,
				 nHeads : int = 8,
				 nEncoderLayers : int = 6,
				 nDecoderLayers : int = 6,
				 activation : Literal["relu", "gelu", "tanh"] = "relu",
				 dropout : float = 0.1,
				 dimFeedforward : int = 2048,
				 normFirst : bool = False) -> None:
		super().__init__()

		self.dModel = dModel
		self.nHeads = nHeads
		self.nEncoderLayers = nEncoderLayers
		self.nDecoderLayers = nDecoderLayers
		self.activation = activation
		self.dropout = dropout
		self.dimFeedforward = dimFeedforward
		self.normFirst = normFirst

		encoderLayer = EncoderLayer(dModel, nHeads, activation, dropout, dimFeedforward, normFirst)
		self.encoder = Encoder(encoderLayer, nEncoderLayers)

		decoderLayer = DecoderLayer(dModel, nHeads, activation, dropout, dimFeedforward, normFirst)
		self.decoder = Decoder(decoderLayer, nDecoderLayers)

	def forward(self, src : Tensor, tgt) -> Tensor:
		"""
		:param src: (batch, seqLen, dModel)
		:param tgt: (batch, seqLen, dModel)
		Both src and tgt have shape (batch, seqLen, dModel)

		:return: (batch, seqLen, dModel)
		"""
		encoderOutput = self.encoder.forward(src)

		seqLen = tgt.size(1)
		causalMask = torch.triu(torch.ones(seqLen, seqLen) * -infinity, diagonal=1) # causal mask for decoder
		decoderOutput = self.decoder.forward(tgt, encoderOutput, encoderOutput, causalMask)
		return decoderOutput
