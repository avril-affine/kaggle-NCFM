import os
import numpy as np
import cPickle as pickle
from PIL import Image
from sklearn.cluster import MiniBatchKMeans


def load_data():
    data_dir = os.path.abspath(__file__)
    data_dir = os.path.dirname(os.path.dirname(data_dir))
    data_dir = os.path.join(data_dir, 'data/train')

    data = []
    label = []
    label_name = []
    folders = [x for x in os.listdir(data_dir) if not x.startswith('.')]
    for i, folder in enumerate(folders):
        for img_name in os.listdir(os.path.join(data_dir, folder)):
            if img_name.startswith('.'):
                continue

            img_path = os.path.join(data_dir, folder, img_name)
            img = Image.open(img_path)
            img = np.array(img.resize((299, 299)))
            img = img.flatten()
            data.append(img)
            label.append(i)
            label_name.append(folder)

    return np.array(data), np.array(label), np.array(label_name)


def main():
    data, label, label_name = load_data()

    model = MiniBatchKMeans(n_clusters=12, verbose=True)
    preds = model.fit_predict(data)

    clusters = [[0 for _ in xrange(8)] for _ in xrange(12)]
    for label, pred in zip(label, preds):
        clusters[pred][label] += 1

    for i in xrange(12):
        print i, clusters[i]

    with open('kmeans.pkl', 'w') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    main()
