from src.model.dataset import GRIDDataset

dataset = GRIDDataset("data/processed")

print("Dataset size:", len(dataset))

# Test first sample
lips, label = dataset[0]

print("\nLip tensor shape:")
print(lips.shape)

print("\nLabel tensor:")
print(label)

print("\nLabel length:")
print(len(label))