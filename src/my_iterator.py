import os
import json
import numpy as np
from keras.preprocessing.image import load_img, img_to_array, DirectoryIterator


class MyDirectoryIterator(DirectoryIterator):
    '''DirectoryIterator to include bounding box outputs.

    # Arguments
        class_mode: Added 'bounding_box' option for bounding box outputs.
        bbox_dict: Required if class_mode 'bounding_box' specified.
            Dict mapping base filenames to bounding boxes.
        img_info: dict containing map from img_name to height/width.
        localizer: Model that predicts a bounding box.
        num_test: How many times to run localizer and average.
    '''

    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256), color_mode='rgb',
                 dim_ordering='default',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg',
                 follow_links=False,
                 bbox_dict=None,
                 img_info=None,
                 localizer=None,
                 num_test=1):
        super(MyDirectoryIterator, self).__init__(
            directory, image_data_generator,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            dim_ordering=dim_ordering,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir, save_prefix=save_prefix,
            save_format=save_format, follow_links=follow_links)

        if class_mode == 'bounding_box':
            if not bbox_dict:
                raise Exception('bbox_dict must be specified for '
                                'class_mode: "bounding_box"')
        if localizer:
            if not img_info:
                raise Exception('img_info must be specified along '
                                'with localizer')

        self.bbox_dict = bbox_dict
        self.img_info  = img_info
        self.localizer = localizer
        self.num_test  = num_test

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
                           grayscale=grayscale)

            img_ = img.resize((self.target_size[1], self.target_size[0]))

            # cropped images
            if self.localizer:
                X = np.zeros((self.num_test,) + self.image_shape)
                resize = np.zeros(4)
                resize[[1, 3]] = self.img_info[fname]['size'][0] / self.target_size[0]
                resize[[0, 2]] = self.img_info[fname]['size'][1] / self.target_size[1]
                for i in xrange(self.num_test):
                    x = img_to_array(img_, dim_ordering=self.dim_ordering)
                    x = self.image_data_generator.random_transform(x)
                    x = self.image_data_generator.standardize(x)
                    X[i] = x
                pred_boxes = self.localizer.predict(X)
                pred_boxes = preds.mean(axis=0)
                pred_boxes *= resize
                pred_boxes = np.maximum(pred_boxes, 0)
                pred_boxes[2] += pred_boxes[0]
                pred_boxes[3] += pred_boxes[1]
                pred_boxes[2:] = np.maximum(pred_boxes[2:], pred_boxes[:2])
                pred_boxes[[0, 2]] = np.minimum(pred_boxes[[0, 2]], self.img_info[fname]['size'][0])
                pred_boxes[[1, 3]] = np.minimum(pred_boxes[[1, 3]], self.img_info[fname]['size'][1])
                pred_boxes = pred_boxes.astype(int)
                x = img.crop((pred_boxes[0], pred_boxes[1], pred_boxes[2], pred_boxes[3]))
                x = x.resize((self.target_size[1], self.target_size[0]))
            else:
                x = img_to_array(img_, dim_ordering=self.dim_ordering)
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
        elif self.class_mode == 'bounding_box':
            filenames = [self.filenames[i] for i in index_array]
            basenames = [os.path.basename(x) for x in filenames]
            batch_y = np.zeros((current_batch_size, 4), dtype='float32')
            for i, name in enumerate(basenames):
                batch_y[i] = self.boxes[name]
        else:
            return batch_x
        return batch_x, batch_y
