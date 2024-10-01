from os.path import join

CHECKPOINT_PATH = "model_checkpoints"
ANN_MODEL_PATH = join(CHECKPOINT_PATH, "ann")
LSTM_MODEL_PATH = join(CHECKPOINT_PATH, "LSTM")
TRANSFORMER_MODEL_PATH = join(CHECKPOINT_PATH, "transformer")

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
EOS_TOKEN = "<EOS>"
SOS_TOKEN = "<SOS>"

def getAnnModelName() -> str:
	pass

def getRnnModelName() -> str:
	pass

def getTransformerModelName() -> str:
	pass
