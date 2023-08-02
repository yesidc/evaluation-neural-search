import os
import random
import json
import shutil

def select_random_files(source_dir, destination_dir, num_files):
    # Get a list of all files in the source directory
    file_list = os.listdir(source_dir)

    # Shuffle the file list
    random.shuffle(file_list)

    # Select the first 'num_files' files
    selected_files = file_list[:num_files]

    # Create the destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    # Copy the selected files to the destination directory
    for file_name in selected_files:
        source_path = os.path.join(source_dir, file_name)
        destination_path = os.path.join(destination_dir, file_name)
        shutil.copyfile(source_path, destination_path)

    # Save the names of the selected files to a JSON file
    json_data = {
        'selected_files': selected_files
    }
    json_file_path = os.path.join(destination_dir, 'selected_image_files_latency.json')
    with open(json_file_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)



select_random_files('/Users/yesidcano/Downloads/val2014',
                    '/Users/yesidcano/Downloads/latency_images_1000',
                    1000)
