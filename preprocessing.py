import cv2
import os
import glob

# Preprocess Wider Face Dataset (minimal images and grayscale)
def preprocess_wider_face(input_dir, output_dir, limit_per_category=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    category_dirs = os.listdir(input_dir)
    for category in category_dirs:
        category_path = os.path.join(input_dir, category)
        if os.path.isdir(category_path):
            image_paths = glob.glob(f"{category_path}/*.jpg")[:limit_per_category]
            output_category_dir = os.path.join(output_dir, category)
            if not os.path.exists(output_category_dir):
                os.makedirs(output_category_dir)

            for img_path in image_paths:
                img = cv2.imread(img_path)
                processed_img = cv2.resize(img, (64, 64))  # Very low resolution
                processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                output_path = os.path.join(output_category_dir, os.path.basename(img_path))
                cv2.imwrite(output_path, processed_img)

# Preprocess CelebA Dataset (minimal images and grayscale)
def preprocess_celeba(input_dir, output_dir, partition_file, limit=100):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(output_dir, split)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
    
    partition_map = {}
    with open(partition_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            img_name = parts[0]
            partition = int(parts[1])
            if partition == 0:
                partition_map[img_name] = 'train'
            elif partition == 1:
                partition_map[img_name] = 'val'
            else:
                partition_map[img_name] = 'test'

    processed_count = { 'train': 0, 'val': 0, 'test': 0 }
    image_paths = os.listdir(input_dir)[:limit]
    for img_name in image_paths:
        img_path = os.path.join(input_dir, img_name)
        if img_name in partition_map and processed_count[partition_map[img_name]] < limit:
            img = cv2.imread(img_path)
            processed_img = cv2.resize(img, (64, 64))  # Very low resolution
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            output_path = os.path.join(output_dir, partition_map[img_name], img_name)
            cv2.imwrite(output_path, processed_img)
            processed_count[partition_map[img_name]] += 1

if __name__ == "__main__":
    preprocess_wider_face("data/wider_face/WIDER_train/images", "data/processed/wider_face")
    preprocess_celeba("data/celeba/img_align_celeba", "data/processed/celeba", "data/celeba/list_eval_partition.txt")