import os
import pickle
import numpy as np
from PIL import Image
from matplotlib import cm

def process_and_save_image(data, file_name, min_length):
    data = data[:min_length]
    right = left = 33
    top = bottom = 14

    # Extract emotion and train/test from filename
    emotion = file_name.split('_')[1]
    train_or_test = file_name.split('_')[2].split(".")[0]

    # Create directories for emotion and train/test
    emotion_dir = f'/home/u956278/EEG_syntethic_data/data/stft/pickle_to_images_stft/{emotion}'
    train_test_dir = f'/home/u956278/EEG_syntethic_data/data/stft/pickle_to_images_stft/{emotion}/{train_or_test}'
    os.makedirs(emotion_dir, exist_ok=True)
    os.makedirs(train_test_dir, exist_ok=True)

    # Process and save each image
    for idx, i in enumerate(data):
        image = Image.fromarray(np.uint8(cm.gist_earth(i)*255))
        width, height = image.size
        new_image = Image.new('RGB', (width + right + left, height + top + bottom), (255, 255, 255))
        new_image.paste(image, (left, top))

        # Save the image with appropriate file name and directory path
        destination_path = f'{train_test_dir}/{emotion}_{idx}.png'
        new_image.save(destination_path)

        print(f"Saved image {idx} from file {file_name} in directory {train_test_dir}.")

# Directory containing the pickle files
dir_path = '/home/u956278/EEG_syntethic_data/data/stft/EEG_to_pickle_stft/'

# Get a list of all pickle files in the directory
pickle_files = [f for f in os.listdir(dir_path) if f.endswith('.pickle')]

# Initialize the minimum length for train and test data
min_length_train = float('inf')
min_length_test = float('inf')

# Store all neural data in a dictionary
neural_data_dict = {}

# First loop: Load all datasets and find the minimum length
for file_name in pickle_files:
    file_path = os.path.join(dir_path, file_name)

    # Load the data from the pickle file
    with open(file_path, 'rb') as f:
        neural_data = pickle.load(f)

    # Update the minimum length for train or test data separately
    train_or_test = file_name.split('_')[2].split(".")[0]
    if train_or_test == "train":
        min_length_train = min(min_length_train, len(neural_data))
    elif train_or_test == "test":
        min_length_test = min(min_length_test, len(neural_data))

    # Store the neural data in the dictionary
    neural_data_dict[file_name] = neural_data

    print(f"Loaded file {file_name} with {len(neural_data)} data points.")

# Second loop: Process and save images
for file_name, neural_data in neural_data_dict.items():
    train_or_test = file_name.split('_')[2].split(".")[0]
    if train_or_test == "train":
        process_and_save_image(neural_data, file_name, min_length_train)
    elif train_or_test == "test":
        process_and_save_image(neural_data, file_name, min_length_test)

print("Done processing and saving all images.")
