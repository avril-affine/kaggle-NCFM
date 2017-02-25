import os
import json
import numpy as np
from keras.preprocessing.image import load_img, img_to_array, DirectoryIterator


class MyDirectoryIterator(DirectoryIterator):

    def __init__(self, directory, image_data_generator, box_file=None,
                 localizer=None, img_info=None, target_size=(256, 256),
                 color_mode='rgb', dim_ordering='default',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg',
                 follow_links=False, box_model=False):
        if box_file:
            with open(box_file, 'r') as f:
                self.boxes = json.loads(f.read())
        self.localizer = localizer
        self.img_info = img_info
        self.box_model = box_model
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
                           grayscale=grayscale)

            # cropped images
            if self.localizer:
                pred_boxes = self.localizer.predict(
                        img.resize((self.target_size[1], self.target_size[0])))
                resize = np.zeros(4)
                resize[[1, 3]] = img_info[name]['size'][0]
                resize[[0, 2]] = img_info[name]['size'][1]
                resize /= 299.          # input size
                pred_boxes[i] *= resize
                x = img.crop((pred_boxes[0],
                              pred_boxes[1],
                              pred_boxes[0] + pred_boxes[2],
                              pred_boxes[1] + pred_boxes[3]))
            else:
                img = img.resize((self.target_size[1], self.target_size[0]))
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

            # bounding box labels
            if self.boxes:
                filenames = np.array(self.filenames)[index_array]
                basenames = [os.path.basename(x) for x in filenames]
                batch_y_box = np.zeros((len(batch_x), 4), dtype='float32')
                for i, name in enumerate(basenames):
                    batch_y_box[i] = self.boxes[name]
                batch_y = [batch_y_box, batch_y]

            if self.box_model:
                batch_y = batch_y_box
        else:
            return batch_x
        return batch_x, batch_y

