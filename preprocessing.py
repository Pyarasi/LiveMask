import cv2
import os
import glob
import random

def preprocess_wider_face(input_dir, output_dir, limit_per_category=10):
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    category_dirs = os.listdir(input_dir)
    for category in category_dirs:
        category_path = os.path.join(input_dir, category)
        if os.path.isdir(category_path):
            image_paths = glob.glob(f"{category_path}/*.jpg")
            if len(image_paths) > limit_per_category:
                image_paths = random.sample(image_paths, limit_per_category) 

            split_index = int(0.8 * len(image_paths))
            train_images = image_paths[:split_index]
            val_images = image_paths[split_index:]

            for img_path in train_images:
                img = cv2.imread(img_path)
                processed_img = cv2.resize(img, (64, 64)) 
                processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY) 
                output_path = os.path.join(train_dir, os.path.basename(img_path))
                cv2.imwrite(output_path, processed_img)

            for img_path in val_images:
                img = cv2.imread(img_path)
                processed_img = cv2.resize(img, (64, 64))
                processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
                output_path = os.path.join(val_dir, os.path.basename(img_path))
                cv2.imwrite(output_path, processed_img)

if __name__ == "__main__":
    preprocess_wider_face("data/wider_face/WIDER_train/images", "data/processed/wider_face")