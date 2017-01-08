import os
import sys
import json


def main():
    if len(sys.argv) != 1:
        print 'python compile_labels.py /path/to/label_directory'
        sys.exit(1)

    label_dir = sys.argv[1]
    label_files = ['alb_labels.json',
                   'bet_labels.json',
                   'dol_labels.json',
                   'lag_labels.json',
                   'nof_labels.json',
                   'other_labels.json',
                   'shark_labels.json',
                   'yft_labels.json']

    compiled = {}
    for label_file in label_files:
        label_file_path = os.path.join(label_dir, label_file)
        with open(label_file_path, 'r') as f:
            boxes = json.loads(f.read())
        if len(boxes['annotations']) == 1:
            compiled[boxes['filename']] = boxes['annotations'][0]
        else:
            print len(boxes['annotaions']), label_file, boxes['filename']

    compiled_path = os.path.join(label_dir, 'compiled.json')
    with open(compiled_path, 'w') as f:
        f.write(json.dumps(compiled)


if __name__ == '__main__':
    main()
