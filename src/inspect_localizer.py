import os
import argparse
import numpy as np
from keras.models import load_model
from PIL import ImageDraw


def main(args):
    model = load_model(args.import_model)

    val_dir = args.val_dir
    val_imgs = os.listdir(val_dir)
    np.random.shuffle(va_imgs)

    for val_img in val_imgs:
        img = load_img(os.path.join(val_dir, val_img), target_size=(299, 299))

        pred = model.predict(img)

        draw = ImageDraw(img)
        draw.rectangle((pred[0],
                        pred[1],
                        pred[0] + pred[2],
                        pred[1] + pred[3]))

        img.show()


if __name__ == '__main__':
    parser.add_argument('--import_model', required=True,
        help='Path to localizer .h5 weights file.')
    parser.add_argument('--val_dir', required=True,
        help='Path to test image files.')

    args = parser.parse_args()

    main(args)
