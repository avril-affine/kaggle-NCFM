from keras.callbacks import TensorBoard


class BatchTensorboard(TensorBoard):
    def __init__(self, log_dir='./logs', histogram_freq=0, write_graph=True,
                 write_images=False, batch_freq=1):
        super(BatchTensorboard, self).__init__(log_dir,
                                                histogram_freq,
                                                write_graph,
                                                write_images)
        self.batch_freq = batch_freq
        self.prev_batch = 0
        self.real_batch = 0

    def on_epoch_end(self, epoch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        import tensorflow as tf

        if batch == 0:
            self.real_batch = self.prev_batch
        batch += self.real_batch
        self.prev_batch = batch

        if self.model.validation_data and batch % self.batch_freq == 0:
            if self.model.uses_learning_phase:
                cut_v_data = len(self.model.inputs)
                val_data = self.model.validation_data[:cut_v_data] + [0]
                tensors = self.model.inputs + [K.learning_phase()]
            else:
                val_data = self.model.vaidation_data
                tensors = self.model.inputs
            feed_dict = dict(zip(tensors, val_data))
            result = self.sess.run([self.merged], feed_dict=feed_dict)
            summary_str = results[0]
            self.writer.add_summary(summary_str, batch)

        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.writer.add_summary(summary, batch)
        self.writer.flush()
