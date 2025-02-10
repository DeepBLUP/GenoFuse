import csv


def convert_raw_to_csv(raw_file_path, csv_file_path):
    with open(raw_file_path, 'r', encoding='utf-8') as raw_file:
        lines = raw_file.readlines()

    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)

        header = lines[0].strip().split()
        csv_writer.writerow(header)

        for line in lines[1:]:
            fields = line.strip().split()
            csv_writer.writerow(fields)

    print(f"completely: {csv_file_path}")


if __name__ == "__main__":
    raw_file_path = r'testdata.raw'
    csv_file_path = r'testdata.csv'
    convert_raw_to_csv(raw_file_path, csv_file_path)
