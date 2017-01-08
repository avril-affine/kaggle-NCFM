import os
import shutil
from sklearn.model_selection import KFold


kf = KFold(n_splits=10, shuffle=True, random_state=0)

folders = [x for x in os.listdir('data/train') if not x.startswith('.')]

for folder in folders:
    imgs = os.listdir('data/train/' + folder)
    imgs = [x for x in imgs if x.endswith('.jpg')]

    for i, (train_index, val_index) in enumerate(kf.split(imgs)):
        train_dir = 'data/train_folds/fold_{0}/train/{1}'.format(i, folder)
        val_dir = 'data/train_folds/fold_{0}/val/{1}'.format(i, folder)
        os.makedirs(train_dir)
        os.makedirs(val_dir)

        train_imgs = [imgs[i] for i in train_index]
        val_imgs = [imgs[i] for i in val_index]
        for img in train_imgs:
            src_path = os.path.join('data/train', folder, img)
            shutil.copy(src_path, train_dir)
        for img in val_imgs:
            src_path = os.path.join('data/train', folder, img)
            shutil.copy(src_path, val_dir)
