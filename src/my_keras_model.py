import warnings
import time
import keras.callbacks as cbks
from keras.models import Model as Mdl
from keras.engine.training import generator_queue


class Model(Mdl):
   def fit_generator(self, generator, samples_per_epoch, nb_epoch,
                      verbose=1, callbacks=None,
                      validation_data=None, nb_val_samples=None,
                      class_weight=None,
                      max_q_size=10, nb_worker=1, pickle_safe=False,
                      initial_epoch=0, validate_batch=True):
        '''Fits the model on data generated batch-by-batch by
        a Python generator.
        The generator is run in parallel to the model, for efficiency.
        For instance, this allows you to do real-time data augmentation
        on images on CPU in parallel to training your model on GPU.

        # Arguments
            generator: a generator.
                The output of the generator must be either
                - a tuple (inputs, targets)
                - a tuple (inputs, targets, sample_weights).
                All arrays should contain the same number of samples.
                The generator is expected to loop over its data
                indefinitely. An epoch finishes when `samples_per_epoch`
                samples have been seen by the model.
            samples_per_epoch: integer, number of samples to process before
                going to the next epoch.
            nb_epoch: integer, total number of iterations on the data.
            verbose: verbosity mode, 0, 1, or 2.
            callbacks: list of callbacks to be called during training.
            validation_data: this can be either
                - a generator for the validation data
                - a tuple (inputs, targets)
                - a tuple (inputs, targets, sample_weights).
            nb_val_samples: only relevant if `validation_data` is a generator.
                number of samples to use from validation generator
                at the end of every epoch.
            class_weight: dictionary mapping class indices to a weight
                for the class.
            max_q_size: maximum size for the generator queue
            nb_worker: maximum number of processes to spin up
                when using process based threading
            pickle_safe: if True, use process based threading.
                Note that because
                this implementation relies on multiprocessing,
                you should not pass
                non picklable arguments to the generator
                as they can't be passed
                easily to children processes.
            initial_epoch: epoch at which to start training
                (useful for resuming a previous training run)

        # Returns
            A `History` object.

        # Example

        ```python
            def generate_arrays_from_file(path):
                while 1:
                    f = open(path)
                    for line in f:
                        # create numpy arrays of input data
                        # and labels, from each line in the file
                        x1, x2, y = process_line(line)
                        yield ({'input_1': x1, 'input_2': x2}, {'output': y})
                    f.close()

            model.fit_generator(generate_arrays_from_file('/my_file.txt'),
                                samples_per_epoch=10000, nb_epoch=10)
        ```
        '''
        wait_time = 0.01  # in seconds
        epoch = initial_epoch

        do_validation = bool(validation_data)
        self._make_train_function()
        if do_validation:
            self._make_test_function()

        # python 2 has 'next', 3 has '__next__'
        # avoid any explicit version checks
        val_gen = (hasattr(validation_data, 'next') or
                   hasattr(validation_data, '__next__'))
        if val_gen and not nb_val_samples:
            raise ValueError('When using a generator for validation data, '
                             'you must specify a value for "nb_val_samples".')

        out_labels = self.metrics_names
        callback_metrics = out_labels + ['val_' + n for n in out_labels]

        # prepare callbacks
        self.history = cbks.History()
        callbacks = [cbks.BaseLogger()] + (callbacks or []) + [self.history]
        if verbose:
            callbacks += [cbks.ProgbarLogger()]
        callbacks = cbks.CallbackList(callbacks)

        # it's possible to callback a different model than self:
        if hasattr(self, 'callback_model') and self.callback_model:
            callback_model = self.callback_model
        else:
            callback_model = self
        callbacks._set_model(callback_model)
        callbacks._set_params({
            'nb_epoch': nb_epoch,
            'nb_sample': samples_per_epoch,
            'verbose': verbose,
            'do_validation': do_validation,
            'metrics': callback_metrics,
        })
        callbacks.on_train_begin()

        if do_validation and not val_gen:
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data
            else:
                raise ValueError('validation_data should be a tuple '
                                 '(val_x, val_y, val_sample_weight) '
                                 'or (val_x, val_y). Found: ' +
                                 str(validation_data))
            val_x, val_y, val_sample_weights = self._standardize_user_data(
                val_x, val_y, val_sample_weight)
            self.validation_data = val_x + [val_y, val_sample_weights]
        else:
            self.validation_data = None

        # start generator thread storing batches into a queue
        data_gen_queue, _stop, generator_threads = generator_queue(
            generator,
            max_q_size=max_q_size,
            nb_worker=nb_worker,
            pickle_safe=pickle_safe)

        callback_model.stop_training = False
        while epoch < nb_epoch:
            callbacks.on_epoch_begin(epoch)
            samples_seen = 0
            batch_index = 0
            while samples_seen < samples_per_epoch:
                generator_output = None
                while not _stop.is_set():
                    if not data_gen_queue.empty():
                        generator_output = data_gen_queue.get()
                        break
                    else:
                        time.sleep(wait_time)

                if not hasattr(generator_output, '__len__'):
                    _stop.set()
                    raise ValueError('output of generator should be a tuple '
                                     '(x, y, sample_weight) '
                                     'or (x, y). Found: ' +
                                     str(generator_output))
                if len(generator_output) == 2:
                    x, y = generator_output
                    sample_weight = None
                elif len(generator_output) == 3:
                    x, y, sample_weight = generator_output
                else:
                    _stop.set()
                    raise ValueError('output of generator should be a tuple '
                                     '(x, y, sample_weight) '
                                     'or (x, y). Found: ' +
                                     str(generator_output))
                # build batch logs
                batch_logs = {}
                if isinstance(x, list):
                    batch_size = x[0].shape[0]
                elif isinstance(x, dict):
                    batch_size = list(x.values())[0].shape[0]
                else:
                    batch_size = x.shape[0]
                batch_logs['batch'] = batch_index
                batch_logs['size'] = batch_size
                callbacks.on_batch_begin(batch_index, batch_logs)

                try:
                    outs = self.train_on_batch(x, y,
                                               sample_weight=sample_weight,
                                               class_weight=class_weight)
                except:
                    _stop.set()
                    raise

                if not isinstance(outs, list):
                    outs = [outs]
                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o

                if validate_batch:
                    try:
                        val_outs = self.evaluate_generator(
                            validation_data,
                            batch_size,
                            max_q_size=max_q_size,
                            nb_worker=nb_worker,
                            pickle_safe=pickle_safe)
                    except:
                        _stop.set()
                        raise
                    if not isinstance(val_outs, list):
                        outs = [outs]
                    for l, o in zip(out_labels, val_outs):
                        if l.startswith('val_'):
                            batch_logs[l] = o

                callbacks.on_batch_end(batch_index, batch_logs)

                # construct epoch logs
                epoch_logs = {}
                batch_index += 1
                samples_seen += batch_size

                # epoch finished
                if samples_seen > samples_per_epoch:
                    warnings.warn('Epoch comprised more than '
                                  '`samples_per_epoch` samples, '
                                  'which might affect learning results. '
                                  'Set `samples_per_epoch` correctly '
                                  'to avoid this warning.')
                if samples_seen >= samples_per_epoch and do_validation:
                    if val_gen:
                        val_outs = self.evaluate_generator(
                            validation_data,
                            nb_val_samples,
                            max_q_size=max_q_size,
                            nb_worker=nb_worker,
                            pickle_safe=pickle_safe)
                    else:
                        # no need for try/except because
                        # data has already been validated
                        val_outs = self.evaluate(
                            val_x, val_y,
                            batch_size=batch_size,
                            sample_weight=val_sample_weights,
                            verbose=0)
                    if not isinstance(val_outs, list):
                        val_outs = [val_outs]
                    # same labels assumed
                    for l, o in zip(out_labels, val_outs):
                        epoch_logs['val_' + l] = o

            callbacks.on_epoch_end(epoch, epoch_logs)
            epoch += 1
            if callback_model.stop_training:
                break

        _stop.set()
        if pickle_safe:
            # Terminate all daemon processes
            for p in generator_threads:
                if p.is_alive():
                    p.terminate()
            data_gen_queue.close()
        callbacks.on_train_end()
        return self.history
