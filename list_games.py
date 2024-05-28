import os


def list_files_in_folder(folder_path):
    files = os.listdir(folder_path)
    file_names_without_extensions = [os.path.splitext(file)[0] for file in files]
    return file_names_without_extensions


folder_path = 'basketballPlayOptimisation/Tracking_Data/game logs'
file_names = list_files_in_folder(folder_path)

# Write the file names to a text file

with open('allgames.txt', 'w') as file:
    for name in file_names:
        file.write(name + '\n')




