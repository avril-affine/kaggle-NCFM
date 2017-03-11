import parser


def base_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--home_dir', required=True,
        help='Path to the root directory of the project')
    parser.add_argument(
        '--data_dir', required=True,
        help='Path to a data directory. If kfolds are specified, '
        'the directory should contain a directory for each fold 0 to k-1, '
        'else it should contain a train and val folder.')
    parser.add_argument(
        '--kfolds', type=int, default=0,
        help='Number of folds to run the model on. The directory structure '
             'must be setup before running this file.')
    parser.add_argument(
        '--fold_prefix', default='fold_',
        help='Prefix for each fold directory.')
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size for the model.')
    parser.add_argument(
        '--shear', type=float, default=0.,
        help='Data augmentation: shear intensity in radians.')
    parser.add_argument(
        '--zoom', type=float, default=0.,
        help='Data augmentation: range for random zoom.')
    parser.add_argument(
        '--rotation', type=float, default=0.,
        help='Data augmentation: range for random rotation in degrees.')
    parser.add_argument(
        '--shift', type=float, default=0.,
        help='Data augmentation: range for random shifts in x/y directions.')
    parser.add_argument(
        '--flip_lr', action='store_true',
        help='Data augmentation: specify to apply random horizontal flips.')
    parser.add_argument(
        '--flip_ud', action='store_true',
        help='Data augmentation: specify to apply random vertical flips.')

    return parser


def train_arguments(parser):
    parser.add_argument(
        '--model_name', required=True,
        help='Name of the model. This will also be the name of the directory '
             'that stores the model weights and run data.')
    parser.add_argument(
        '--import_model', required=True,
        help='Model to import and run in the form of '
             'path.to.script.function: where `path.to.script` is the module '
             'to import and `function` is the function within the module.')
    parser.add_argument(
        '--learning_rate', type=float, default=0.0001,
        help='Learning rate of the model.')
    parser.add_argument(
        '--epochs', type=int, default=10,
        help='Number of epochs to run the model.')
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size for the model.')
    parser.add_argument(
        '--fine_tune', action='store_true',
        help='Only retrain last layers.')

    return parser
