import os
import shutil

# Define source and destination directories
source_root = 'sketch'
destination_folder = 'data'

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Supported image extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')

# Counter for renaming
counter = 1

# Traverse all subdirectories
for subdir, _, files in os.walk(source_root):
    for file in files:
        if file.lower().endswith(image_extensions):
            # Construct full source path
            src_path = os.path.join(subdir, file)
            
            # Create new filename
            ext = os.path.splitext(file)[1]
            new_name = f"sketch_{counter:05d}{ext}"
            dst_path = os.path.join(destination_folder, new_name)
            
            # Copy and rename
            shutil.copy2(src_path, dst_path)
            counter += 1

print(f"Copied and renamed {counter - 1} images to '{destination_folder}'.")
