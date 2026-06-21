import numpy as np
import matplotlib.pyplot as plt

data = np.load("data/processed/bbaf2n.npy")

print("Shape:", data.shape)

plt.imshow(data[0], cmap="gray")
plt.title("First Lip Frame")
plt.show()