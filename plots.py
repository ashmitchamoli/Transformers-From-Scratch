from matplotlib import pyplot as plt

# EFFECT OF NUMBER OF LAYERS #
model1 = { # nLayers = 1
	"Train Loss": [6.515, 5.000, 3.949, 3.200, 2.645, 2.417, 2.227, 2.082],
	"Val Loss": [6.092, 5.048, 4.253, 3.678, 3.167, 2.853, 2.640, 2.486],
	"Val Accuracy": [21.152, 38.984, 49.327, 57.408, 62.964, 66.418, 69.321, 71.398]
}

model2 = { # nLayers = 3
	"Train Loss": [6.364, 4.767, 3.671, 2.932, 2.373, 2.124, 1.941],
	"Val Loss": [5.915, 4.713, 3.924, 3.300, 2.813, 2.492, 2.319],
	"Val Accuracy": [22.892, 43.000, 53.670, 61.428, 67.159, 71.125, 73.803]
}


for key in model1:
	plt.title(key)
	plt.plot(model1[key], label="nLayers = 1")
	plt.plot(model2[key], label="nLayers = 3")
	plt.xticks(range(1, max(len(model1[key]), len(model2[key])) + 1))
	plt.legend()
	plt.savefig(f"plots/{key}_nLayers.png")
	plt.show()
