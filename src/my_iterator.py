import os
import json
import numpy as np
from keras.preprocessing.image import load_img, img_to_array, DirectoryIterator


class MyDirectoryIterator(DirectoryIterator):

    def __init__(self, box_file, directory, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 dim_ordering='default',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg',
                 follow_links=False):
        with open(box_file, 'r') as f:
            self.boxes = json.loads(f.read())
        super(MyDirectoryIterator, self).__init__(
            directory, image_data_generator,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            dim_ordering=dim_ordering,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir, save_prefix=save_prefix,
            save_format=save_format, follow_links=follow_links)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros((current_batch_size,) + self.image_shape)
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = load_img(os.path.join(self.directory, fname),
                           grayscale=grayscale,
                           target_size=self.target_size)
            x = img_to_array(img, dim_ordering=self.dim_ordering)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype('float32')
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.nb_class), dtype='float32')
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
            filenames = np.array(self.filenames)[index_array]
            basenames = [os.path.basename(x) for x in filenames]
            batch_y_box = np.zeros((len(batch_x), 4), dtype='float32')
            for i, name in enumerate(basenames):
                batch_y_box[i] = self.boxes[name]
        else:
            return batch_x
        return batch_x, [batch_y_box, batch_y]


# class LCGenerator(object):
# 
#     def __init__(self, img_dir, box_file, batch_size, target_size,
#                  shuffle=False):
#         """Localize and Classify Generator.
#         
#         Arguments:
#             img_dir (str): Path to folders containing images with each folder
#                 corresponding to a label.
#             box_file (str): Json file containing bounding boxes for each image.
#             batch_size (int): Batch size.
#             target_size (tuple): Size of returned images (width, height).
#         """
#         self.imgs, self.labels = self._parse_dirs(img_dir)
#         with open(box_file, 'r') as f:
#             boxes = json.loads(f.read())
#         self.boxes = [boxes[os.path.basename(x)] for x in self.train_imgs]
#         self.batch_size = batch_size
#         self.target_size = target_size
#         self.shuffle = shuffle
#         self.N = len(self.imgs)
#         self.current_index = 0
#         self.index_array = np.arange(self.N)
# 
#     def _parse_dirs(self, img_dir):
#         imgs = []
#         labels = []
#         
#         folders = [x for x in os.listdir(img_dir) if not x.startswith('.')]
#         for folder in folders:
#             folder_path = os.path.join(img_dir, folder)
#             img_paths = [os.path.join(folder_path, x)
#                          for x in os.listdir(folder_path)
#                          if x.endswith('.jpg')]
#             imgs.extend(img_paths)
#             labels.extend([folder] * len(img_paths))
# 
#         return np.array(imgs), np.array(labels)
# 
#     def next(self):
#         if self.shuffle and self.current_index == 0:
#             self.index_array = np.random.permutation(self.index_array)
# 
#         mask = self.index_array[self.current_index:
#                                 self.current_index + self.batch_size]
#         batch_x
