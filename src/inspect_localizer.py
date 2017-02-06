import os
import argparse
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img
from PIL import ImageDraw
from my_keras_model import Model


def main(args):
    model = load_model(args.import_model)
    output = model.get_layer('localize').output
    model = Model(model.input, output)

    val_dir = args.val_dir
    val_imgs = os.listdir(val_dir)
    np.random.shuffle(val_imgs)

    for val_img in val_imgs:
        img = load_img(os.path.join(val_dir, val_img), target_size=(299, 299))

        pred = model.predict(np.expand_dims(np.array(img), axis=0))[0]

        draw = ImageDraw.Draw(img)
        draw.rectangle((pred[0],
                        pred[1],
                        pred[0] + pred[2],
                        pred[1] + pred[3]),
                       fill=(255, 0, 0))

        img.show()
        print pred[0], pred[1], pred[0] + pred[2], pred[1] + pred[3]
        raw_input('Press Enter to continue...')     # wait for input


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Inspect images from localizer model.')
    parser.add_argument('--import_model', required=True,
        help='Path to localizer .h5 weights file.')
    parser.add_argument('--val_dir', required=True,
        help='Path to test image files.')

    args = parser.parse_args()

    main(args)
