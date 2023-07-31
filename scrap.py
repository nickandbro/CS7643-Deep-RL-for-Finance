import os

root_dir = os.getcwd()  # get the current working directory
output_file = 'output.txt'  # name of output file

# List of target files
target_files = ["main.py", "a2c.py", "env_creation_functions.py", "simple_stock_env.py"]

with open(output_file, 'w') as outfile:
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename in target_files:  # check if the filename is in the target files
                relative_path = os.path.join(dirpath, filename).replace(root_dir + os.sep, '')
                outfile.write('File path: ' + relative_path + '\n')  # write the relative path
                with open(os.path.join(dirpath, filename), 'r', errors='ignore') as a_file:
                    outfile.write(a_file.read())
                    outfile.write('\n\n')  # add two newlines for separation between files
