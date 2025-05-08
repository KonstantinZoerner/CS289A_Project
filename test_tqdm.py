from tqdm import tqdm
import time

# Create the progress bar once

for j in range(10):
    pbar = tqdm(range(100))
    for i in pbar:
        for k in range(10):
            # Update the description of the same progress bar
            pbar.set_description(f"Processing {i, j, k}")
            time.sleep(0.01)