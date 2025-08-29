# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("wanghaohan/imagenetsketch")

# print("Path to dataset files:", path)

import kagglehub, shutil, os

# Download to kagglehub cache
path = kagglehub.dataset_download("wanghaohan/imagenetsketch")

# Move to local folder of your choice
local_path = "dataset"
shutil.copytree(path, local_path, dirs_exist_ok=True)

print("Saved locally at:", os.path.abspath(local_path))
