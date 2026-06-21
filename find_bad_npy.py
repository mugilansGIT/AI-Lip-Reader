import os
import numpy as np

processed_dir = "data/processed"

bad_files = []

for root, dirs, files in os.walk(processed_dir):

    for file in files:

        if not file.endswith(".npy"):
            continue

        path = os.path.join(root, file)

        try:
            arr = np.load(path)

            if len(arr.shape) < 3:
                bad_files.append((path, arr.shape))
                continue

            if arr.shape[0] == 0:
                bad_files.append((path, arr.shape))

        except Exception as e:
            bad_files.append((path, str(e)))

print("\nBAD FILES FOUND:\n")

for item in bad_files:
    print(item)

print("\nTotal bad files:", len(bad_files))