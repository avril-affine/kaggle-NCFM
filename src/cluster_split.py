import os
import numpy as np
import h5py
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
    folders = [x for x in os.listdir(data_dir)
               if not x.startswith('.')
               and os.path.isdir(os.path.join(data_dir, x))]
    for i, folder in enumerate(folders):
        img_folder = os.path.join(data_dir, folder)
        for img_name in os.listdir(img_folder):
            if not img_name.endswith('.jpg'):
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
    if not os.path.exists('data.h5'):
        data, label, label_name = load_data()

        with h5py.File('data.h5', 'w') as hf:
            hf.create_dataset('data', data=data)
            hf.create_dataset('label', data=label)
            hf.create_dataset('label_name', data=label_name)
    else:
        with h5py.File('data.h5', 'r') as hf:
            data = np.array(hf.get('data')[:])
            label = np.array(hf.get('label')[:])
            label_name = np.array(hf.get('label_name')[:])

    print data.shape
    model = MiniBatchKMeans(n_clusters=12, verbose=True)
    preds = model.fit_predict(data)
    with open('kmeans.pkl', 'w') as f:
        pickle.dump(model, f)

    clusters = [[0 for _ in xrange(8)] for _ in xrange(12)]
    for label, pred in zip(label, preds):
        clusters[pred][label] += 1

    for i in xrange(12):
        print i, clusters[i]

    return clusters


if __name__ == '__main__':
    main()
