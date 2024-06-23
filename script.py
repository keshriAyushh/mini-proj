import pandas as pd
import os
import re
import shutil
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Loading the labels.csv
labels_df = pd.read_csv('C:/Users/ayush/OneDrive/Desktop/Solar Panel Fault Detection using Yolov5/labels.csv', header=None, names=['filename', 'probability', 'type'])

# Removing leading/trailing spaces from filenames
labels_df['filename'] = labels_df['filename'].str.replace('images/', '')

# Splitting the data into train and validation sets
train_df, val_df = train_test_split(labels_df, test_size=0.2, random_state=42)

# Define function to create YOLO format labels
def create_yolo_labels(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for _, row in df.iterrows():
        filename = row['filename']
        probability = row['probability']

        # Assuming binary classification: 0 for no defect, 1 for defect
        class_id = 1 if probability > 0 else 0

        # YOLO format: class x_center y_center width height (normalized)
        # Since no bounding box, use entire image (1x1 centered at (0.5, 0.5))
        yolo_format = f"{class_id} 0.5 0.5 1.0 1.0\n"

        # Write to label file
        base_filename = os.path.splitext(os.path.basename(filename))[0]
        base_filename = re.sub(r'\d+$', '', base_filename)
        label_filename = base_filename .replace('.png', '.txt')
        label_path = os.path.join(output_dir, label_filename)
        with open(label_path, 'w') as f:
            f.write(yolo_format)

# Create YOLO format labels
create_yolo_labels(train_df, 'dataset/labels/train')
create_yolo_labels(val_df, 'dataset/labels/val')

# Define the path to the images directory
images_dir = 'C:/Users/ayush/OneDrive/Desktop/Solar Panel Fault Detection using Yolov5/images'

# Function to move images to train and val folders
def move_images(df, subset):
    for filename in df['filename']:
        # Modify the filename to match the actual filenames in the directory
        filename = 'cell' + filename.split('cell')[-1].split('.png')[0] + '.png'
        
        src = os.path.join(images_dir, filename)
        dst = os.path.join(f'dataset/images/{subset}', filename)
        
        # Check if the source file exists before moving
        if os.path.exists(src):
            shutil.move(src, dst)
        else:
            print(f"File not found: {src}")

# Create directories if they do not exist
os.makedirs('dataset/images/train', exist_ok=True)
os.makedirs('dataset/images/val', exist_ok=True)

# Move training and validation images
move_images(train_df, 'train')
move_images(val_df, 'val')

