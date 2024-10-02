import os
import torch
from torch import Tensor, inf as infinity
from torch.nn import Module, Linear, Softmax, Sequential, Tanh, Embedding
from typing import Literal, Optional
from alive_progress import alive_bar as aliveBar
from bidict import bidict

from lib.transformer.encoder import Encoder, EncoderLayer
from lib.transformer.decoder import Decoder, DecoderLayer
from lib.config import PAD_TOKEN

class Transformer(Module):
	def __init__(self,
			  	 vocabularyTgt : bidict,
				 vocabularySrc : bidict,
				 dModel : int = 512,
				 nHeads : int = 8,
				 nEncoderLayers : int = 6,
				 nDecoderLayers : int = 6,
				 activation : Literal["relu", "gelu", "tanh", "elu"] = "relu",
				 dropout : float = 0.1,
				 dimFeedforward : int = 2048,
				 normFirst : bool = False,
				 classifierLayers : list[int] = []) -> None:
		super().__init__()

		self.vocabularyTgt = vocabularyTgt
		self.vocabularySrc = vocabularySrc

		self.dModel = dModel
		self.nHeads = nHeads
		self.nEncoderLayers = nEncoderLayers
		self.nDecoderLayers = nDecoderLayers
		self.activation = activation
		self.dropout = dropout
		self.dimFeedforward = dimFeedforward
		self.normFirst = normFirst

		self.embeddingsTgt = Embedding(len(vocabularyTgt), dModel)
		self.embeddingsSrc = Embedding(len(vocabularySrc), dModel)

		encoderLayer = EncoderLayer(dModel, nHeads, activation, dropout, dimFeedforward, normFirst)
		self.encoder = Encoder(encoderLayer, nEncoderLayers)

		decoderLayer = DecoderLayer(dModel, nHeads, activation, dropout, dimFeedforward, normFirst)
		self.decoder = Decoder(decoderLayer, nDecoderLayers)

		# self.classifier = Linear(dModel, vocabSize)
		classifierLayers = [dModel] + classifierLayers + [len(vocabularyTgt)]
		self.classifier = Sequential()
		for i in range(len(classifierLayers) - 1):
			self.classifier.append(Linear(classifierLayers[i], classifierLayers[i + 1]))
			# self.classifier.append(Tanh())

		self.softmax = Softmax(dim=-1)

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self._modelSaveDir_ = "transformer_ckpt"
		self._modelName_ = f"{self.dModel}_{self.nHeads}_{self.nEncoderLayers}_{self.nDecoderLayers}_{self.activation}_{self.dropout}_{self.dimFeedforward}_{self.normFirst}_{classifierLayers}"
		self._savePath_ = os.path.join(self._modelSaveDir_, self._modelName_ + ".pth")
		print(f"Model will be saved to {self._savePath_}.")

	def _getPositionalEncoding_(self, maxSequenceLength : int, embeddingSize : int, device : torch.device) -> torch.Tensor:
		pe = torch.arange(0, maxSequenceLength, device=device).unsqueeze(1) # (maxSequenceLength, 1)
		temp = torch.arange(0, embeddingSize, device=device).unsqueeze(0) # (1, embeddingSize)
		pe = (pe / 10000 ** ((temp - temp%2) / embeddingSize)) # (maxSequenceLength, embeddingSize)
		pe[:, 0::2] = torch.sin(pe[:, 0::2])
		pe[:, 1::2] = torch.cos(pe[:, 1::2])

		return pe

	def forward1(self, src : Tensor, tgt) -> Tensor:
		"""
		:param src: (batch, seqLen)
		:param tgt: (batch, seqLen)
		Both src and tgt have shape (batch, seqLen)

		:return: (batch, seqLen, dModel)
		"""
		src = self.embeddingsSrc(src) + self._getPositionalEncoding_(src.size(1), self.dModel, self.device) # (batch, seqLen, dModel)
		tgt = self.embeddingsTgt(tgt) + self._getPositionalEncoding_(tgt.size(1), self.dModel, self.device) # (batch, seqLen, dModel)

		x = self.encoder.forward(src) # (batch, seqLen, dModel)

		seqLen = tgt.size(1)
		causalMask = torch.triu(torch.ones(seqLen, seqLen) * -infinity, diagonal=1).to(self.device) # causal mask for decoder
		x = self.decoder.forward(tgt, x, x, causalMask) # (batch, seqLen, dModel)
		return x
	
	def forward(self, src : Tensor, tgt) -> Tensor:
		"""
		:param x: (batch, seqLen, dModel)
		
		:return: (batch, seqLen, vocabSize)
		"""
		x = self.forward1(src, tgt)
		return self.classifier(x) # (batch, seqLen, vocabSize)
	
	def trainModel(self,
				   trainLoader : torch.utils.data.DataLoader,
				   valLoader : torch.utils.data.DataLoader,
				   learningRate : float = 1e-3,
				   epochs : int = 1) -> None:
		
		optimizer = torch.optim.Adam(self.parameters(), lr=learningRate)
		criterion = torch.nn.CrossEntropyLoss(ignore_index=self.vocabularyTgt[PAD_TOKEN])

		self.to(self.device)
		for epoch in range(epochs):
			with aliveBar(len(trainLoader)) as bar:
				totalLoss = 0
				for i, (src, tgt) in enumerate(trainLoader):
					src = src.to(self.device)
					tgt = tgt.to(self.device)

					optimizer.zero_grad()
					output = self.forward(src[:, :-1], tgt[:, :-1]).view(-1, len(self.vocabularyTgt)) # (batch * (seqLen-1), vocabSize)

					loss = criterion(output, tgt[:, 1:].reshape(-1))
					loss.backward()
					optimizer.step()

					totalLoss += loss.item()
					bar.text(f"Total Loss: {totalLoss / (i + 1) :.3f}")
					bar()

			with aliveBar(len(valLoader)) as bar:
				totalValLoss = 0
				for i, (src, tgt) in enumerate(valLoader):
					src = src.to(self.device)
					tgt = tgt.to(self.device)

					output = self.forward(src[:, :-1], tgt[:, :-1]).view(-1, len(self.vocabularyTgt)) # (batch * (seqLen-1), vocabSize)

					loss = criterion(output, tgt[:, 1:].reshape(-1))
					totalValLoss += loss.item()
					bar.text(f"Total Loss: {totalValLoss / (i + 1) :.3f}")
					bar()
			
			print(f"Epoch {epoch+1}/{epochs} | Train Loss: {totalLoss / len(trainLoader) :.3f} | Val Loss: {totalValLoss / len(valLoader) :.3f}")
		
		self._saveModel_(self._savePath_)
		
	def _saveModel_(self, path : str) -> None:
		if not os.path.exists(os.path.dirname(path)):
			os.makedirs(os.path.dirname(path))
		torch.save(self.state_dict(), path)
	
	def _loadModel_(self, path : str) -> bool:
		if os.path.exists(path):
			self.load_state_dict(torch.load(path, weights_only=True))
			return True
		else:
			return False

	def loadModelWeights(self, searchPath : Optional[str] = None) -> None:
		searchDir = None
		if searchPath is not None:
			searchDir = os.path.join(searchPath, self._modelSaveDir_)
		else:
			searchDir = self._modelSaveDir_
		if self._loadModel_(os.path.join(searchDir, self._modelName_ + '.pth')):
			print(f"Loaded model from {os.path.join(searchDir, self._modelName_ + '.pth')}")
		else:
			print("Model checkpoint not found. Train the model from scratch.")