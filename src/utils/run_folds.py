import os
import subprocess


def run_folds(args):
    kfolds = args.__dict__.pop('kfolds')
    data_dir = args.__dict__.pop('data_dir')
    fold_prefix = args.__dict__.pop('fold_prefix')
    for fold in xrange(kfolds):
        print '\n------------------Fold %i------------------\n' % fold
        fold_dir = os.path.join(data_dir, fold_prefix, str(fold))
        cmd = ['python', __file__,
               '--data_dir', fold_dir]
        for k, v in args.__dict__.iteritems():
            option = '--' + k
            if isinstance(v, bool):
                if v:
                    cmd += [option]
            else:
                if v is not None:
                    cmd += [option, v]
        subprocess.call(cmd, shell=True)
