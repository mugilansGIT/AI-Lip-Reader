import os

for root, dirs, files in os.walk("data/processed"):

    print(root)

    for f in files[:5]:
        print("   ", f)

    print()