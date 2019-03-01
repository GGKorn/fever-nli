import csv
import os

def main():

    input_file = input('Enter the name of the CSV file you wish to split\nThis must be in the format `filename.csv`\nThis must include a directory if the desired file is in a subfolder of `split_data.py`:')
    output_folder = input('Enter the name of the (already existing) folder you want these split files to appear in\nThis must be in the format `foldername/`:')
    output_prefix = input('Enter your desired prefix for these new files (each one will end with `_x.csv`, with x being their respected indices):')
    split_size = input('How many lines do you want in each new CSV file?\nThis must be an integer:')

    def split_csv(source_filepath, dest_folder, split_file_prefix,
                    records_per_file):
        """
        Split a source csv into multiple csvs of equal numbers of records,
        except the last file (which has however many are remaining from the previous split).

        Includes the initial header row in each split file.

        Split files follow a zero-index sequential naming convention:

            `{split_file_prefix}_0.csv`
        """
        if records_per_file <= 0:
            raise Exception('records_per_file must be > 0')

        if not source_filepath.endswith('.csv'):
            raise Exception('Invalid source file type. Please enter a `.csv` file')

        with open(source_filepath, 'r') as source:
            reader = csv.reader(source)
            headers = next(reader)

            file_index = 0
            records_exist = True

            # Build new files from source_filepath
            while records_exist:
                # i is the index of the number of lines already written in the current file split
                i = 0
                target_filename = f'{split_file_prefix}_{file_index}.csv'
                target_filepath = os.path.join(dest_folder, target_filename)

                with open(target_filepath, 'w') as target:
                    writer = csv.writer(target)

                    while i < records_per_file:
                        # Write header to each new file made
                        if i == 0:
                            writer.writerow(headers)

                        # If we get an error when we try to write from the source_filepath
                        # (i.e. if there is no line from source_filepath left to write),
                        # set records_exist to False and exit this while loop
                        try:
                            writer.writerow(next(reader))
                            i += 1

                        except:
                            records_exist = False
                            break

                # If no lines from source_filepath were written in the last file split, delete this split
                if i == 0:
                    os.remove(target_filepath)

                file_index += 1

    split_csv(input_file, output_folder, output_prefix, int(split_size))

if __name__ == '__main__':
    main()
