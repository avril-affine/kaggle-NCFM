import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans


def load_data():
    data_dir = os.path.abspath(__file__)
    data_dir = os.path.dirname(os.path.dirname(data_dir))
    data_dir = os.path.join(data_dir, 'data/train')

    data = []
    label = []
    label_name = []
    for i, folder in enumerate(os.listdir(data_dir)):
        if folder.startswith('.'):
            continue

        for img_path in os.listdir(os.path.join(data_dir, folder)):
            if img_path.startswith('.'):
                continue

            img = Image.open(img_path)
            img = img.resize((299, 299)).flatten()
            data.append(img)
            label.append(i)
            label_name.append(folder)

    df = pd.DataFrame({'data': data, 'label': label, 'label_name': label_name})
    return df


def main():
    df = load_data()

    model = KMeans(n_clusters=12)
    preds = model.fit_predict(df['data'].values)

    clusters = [[0 for _ in xrange(12)] for _ in xrange(12)]
    for label, pred in zip(df['label'].values, preds):
        clusters[pred][label] += 1

    for i in xrange(12):
        print i, clusters[i]


if __name__ == '__main__':
    main()
