from pathlib import Path
import numpy as np

folder_path = Path(r"D:\University\Summer 26\AICData\clip-features-32")

total_vectors = 0

for file_path in folder_path.glob("*.npy"):
    # Read header only using memory mapping for speed and low RAM usage
    data = np.load(file_path, mmap_mode="r")

    # Assuming the first dimension represents the number of vectors
    if data.ndim >= 1:
        total_vectors += data.shape[0]

print(f"Total vectors: {total_vectors:,}")





