import os


def load_data():
    data_dir = os.path.absos.path.join(__file__, '../data/train')
    print os.listdir(data_dir)


def main():
    load_data()


if __name__ == '__main__':
    main()
