# brats_dataset.py ← MODERN & RELIABLE VERSION (2025+)
import os
import kagglehub

# This is where kagglehub will download and cache the dataset
print("Downloading BraTS2020 dataset using kagglehub (~65 GB)...")
print("This may take 20–90 minutes depending on your internet speed.\n")

# Download the dataset (automatically cached, so second run is instant)
path = kagglehub.dataset_download("awsaf49/brats20-dataset-training-validation")

print("\nDownload complete!")
print("Dataset downloaded and extracted to:")
print("   ", path)

# Now we create a clean, predictable folder at D:\DLH\brats20_download
TARGET_DIR = r"D:\DLH\brats20_download"
os.makedirs(TARGET_DIR, exist_ok=True)

# kagglehub extracts everything into a subfolder — we move everything up one level
import shutil
for item in os.listdir(path):
    src = os.path.join(path, item)
    dst = os.path.join(TARGET_DIR, item)
    if os.path.exists(dst):
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        else:
            os.remove(dst)
    shutil.move(src, dst)

print("\nDataset moved to your project folder:")
print("   ", TARGET_DIR)
print("\nYou can now run:")
print("   python brats_250.py")