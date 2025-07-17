import os, csv

def append_to_csv(filename, class_id, path="predictions.csv"):
    write_header = not os.path.exists(path)

    with open(path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',')
        if write_header:
            writer.writerow(["Filename", "ClassId"])
        writer.writerow([filename, class_id])