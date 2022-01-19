import sys
import os


def convert_labels(file):
    '''Converts the last token on each line to the corresponding regression
    number on a Likert scale of 1-7.'''
    converted_data = []
    with open(file, encoding='utf8') as file:
        data = file.readlines()
    for line in data:
        line = line.rstrip()
        if line[-1] == '1':
            converted_data.append(line[:-1] + '7' + '\n')
        else:
            converted_data.append(line[:-1] + '1' + '\n')
    return converted_data


def write_to_file(data, file_name):
    '''Saves the converted data to a new file in the Regression Task folder
    with the same name as the original file.'''
    directory = os.getcwd() + '/Regression Task'
    with open(os.path.join(directory , file_name), 'w') as file:
        for line in data:
            file.write(line)


def main(argv):
    '''Converts the binary labels to regression labels. Provide as the first
    argument the file you wish to convert. The file should be in the same folder
    as this script.'''
    input_file = argv[1]
    converted_data = convert_labels(input_file)
    write_to_file(converted_data, argv[1])


if __name__ == '__main__':
    main(sys.argv)