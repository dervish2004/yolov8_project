import os
import shutil
from pathlib import Path

# Set your paths
helmet_path = Path("datasets/helemet_dataset")
vehicle_path = Path("datasets/vehicle_dataset")
merged_path = Path("datasets/helmet_vehicle_merged")

# Define subfolders
splits = ['train', 'val', 'test']

# Create merged folder structure
for split in splits:
    (merged_path / 'images' / split).mkdir(parents=True, exist_ok=True)
    (merged_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

def copy_data(src, dst, label_offset=0):
    images_src = src / 'images'
    labels_src = src / 'labels'

    for img_path in images_src.glob('*.jpg'):
        shutil.copy(img_path, dst / 'images' / split / img_path.name)

        # Corresponding label file
        label_path = labels_src / img_path.with_suffix('.txt').name
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
            # Update class IDs (offset for 2nd dataset)
            new_lines = []
            for line in lines:
                if line.strip():
                    parts = line.strip().split()
                    parts[0] = str(int(parts[0]) + label_offset)
                    new_lines.append(" ".join(parts) + "\n")
            with open(dst / 'labels' / split / label_path.name, 'w') as f:
                f.writelines(new_lines)

# Copy helmet (class 0)
for split in splits:
    src = helmet_path / split
    dst = merged_path
    (dst / 'images' / split).mkdir(parents=True, exist_ok=True)
    (dst / 'labels' / split).mkdir(parents=True, exist_ok=True)
    copy_data(src, dst, label_offset=0)

# Copy vehicle (class 1)
for split in splits:
    src = vehicle_path / split
    dst = merged_path
    copy_data(src, dst, label_offset=1)

# Create data.yaml
yaml_text = """
train: datasets/helmet_vehicle_merged/images/train
val: datasets/helmet_vehicle_merged/images/val

nc: 2
names: ['helmet', 'vehicle']
"""

with open(merged_path / 'data.yaml', 'w') as f:
    f.write(yaml_text)

print("✅ Merge complete! YAML created at: helmet_vehicle_merged/data.yaml")
