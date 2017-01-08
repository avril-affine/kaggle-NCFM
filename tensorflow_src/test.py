import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.platform import gfile
from collections import defaultdict


IMG_DIR = 'data/test_stg1'
BOTTLENECK_DIR = 'data/bottlenecks/test/'

BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_INPUT_NAME = 'input/BottleneckInputPlaceholder:0'
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
OUTPUT_TENSOR_NAME = 'final_result:0'

BATCH_SIZE = 1028

def get_bottleneck(sess, filename, fc7_tensor, jpeg_tensor):
    path = os.path.join(BOTTLENECK_DIR, filename)
    if not os.path.exists(path):
        image_path = os.path.join(IMG_DIR, filename)
        image = gfile.FastGFile(image_path, 'rb').read()
        fc7 = sess.run(fc7_tensor, feed_dict={jpeg_tensor: image})
        fc7 = np.squeeze(fc7)
        bottleneck_str = ','.join(str(x) for x in fc7)
        with open(path, 'w') as f:
            f.write(bottleneck_str)

    with open(path, 'r') as f:
        bottleneck_str = f.read()
    return [float(x) for x in bottleneck_str.split(',')]
    

def main():
    if len(sys.argv) != 2:
        print 'Input MODEL_DIR as parameter'
        return

    MODEL_DIR = sys.argv[1]

    with open(MODEL_DIR + 'output_labels.txt', 'r') as f:
        labels = [line.strip() for line in f]

    with tf.Session() as sess:
        with open(MODEL_DIR + 'output_graph.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        sess.graph.as_default()
        fc7_tensor, jpeg_tensor, input_tensor, scores_tensor = (
                tf.import_graph_def(graph_def, 
                                    return_elements=[BOTTLENECK_TENSOR_NAME,
                                                     JPEG_DATA_TENSOR_NAME,
                                                     BOTTLENECK_INPUT_NAME,
                                                     OUTPUT_TENSOR_NAME],
                                    name=''))
        
        img_list = [x for x in os.listdir(IMG_DIR) if x.endswith('.jpg')]
        scores = []
        for i in xrange(0, len(img_list), BATCH_SIZE):
            img_batch = []
            for j in xrange(BATCH_SIZE):
                if i + j == len(img_list):
                    break
                filename = img_list[i+j]
                data = get_bottleneck(sess, filename, fc7_tensor, jpeg_tensor)
                img_batch.append(data)
            
            feed_dict = {input_tensor: img_batch}
            batch_scores = sess.run(scores_tensor, feed_dict=feed_dict)
            scores.extend(batch_scores.tolist())

        labels = [label.upper() if label != 'nof' else 'NoF'
                  for label in labels]
        df = pd.DataFrame(scores, columns=labels)
        df['image'] = img_list
        df.to_csv(MODEL_DIR + 'results.csv', index=False)


if __name__ == '__main__':
    main()
