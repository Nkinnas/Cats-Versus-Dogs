import os
import random
import shutil

# Set the path to your folder containing images
folder_path = "C:/Users/Trevor/cat_dog/cat-v-dog/PetImages/Cat"

# Get a list of all files in the folder
files = os.listdir(folder_path)

# Shuffle the list of files randomly
random.shuffle(files)

# Calculate the split indices
split_index = int(0.8 * len(files))

# Split the files into train and test sets
train_files = files[:split_index]
test_files = files[split_index:]

# Create directories for train and test sets
train_path = "C:/Users/Trevor/cat_dog/cat-v-dog/PetImages/Test/Cat"
test_path = "C:/Users/Trevor/cat_dog/cat-v-dog/PetImages/Train/Cat"

# Move train files to train directory
for file in train_files:
    src = os.path.join(folder_path, file)
    dst = os.path.join(train_path, file)
    shutil.move(src, dst)

# Move test files to test directory
for file in test_files:
    src = os.path.join(folder_path, file)
    dst = os.path.join(test_path, file)
    shutil.move(src, dst)

print("Splitting complete.")
