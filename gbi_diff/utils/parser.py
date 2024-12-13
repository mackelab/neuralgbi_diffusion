from argparse import ArgumentParser


def add_mcmc_sample_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--checkpoint",
        help="path to checkpoint",
        dest="checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--observed-data",
        help="path to observed data.",
        dest="observed_data",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--size",
        help="how many samples you would like to sample. Defaults to 100.",
        dest="size",
        type=int,
        default=100,
        required=False,
    )
    parser.add_argument(
        "--config-file",
        help='path to config file. Defaults to "config/mcmc.yaml".',
        dest="config_file",
        type=str,
        default="config/mcmc.yaml",
        required=False,
    )
    parser.add_argument(
        "--output",
        help="Directory where to store the sampled results. If this is None it will be a subdirectory in the checkpoint directory. Defaults to None",
        dest="output",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--plot",
        help="would like to create a pair-plot with your sampled data. Defaults to False",
        dest="plot",
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--num-worker",
        help="How many threads you would like to use to sample from mcmc",
        dest="num_worker",
        type=int,
        default=1,
        required=False,
    )
    return parser


def add_train_theta_noise_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--config-file",
        help='path to config file (allowed are yaml, toml and json). Defaults to: "config/train.yaml"',
        dest="config_file",
        type=str,
        default="config/train_theta_noise.yaml",
        required=False,
    )
    parser.add_argument(
        "--device",
        help="set to a number to indicate multiple devices. Defaults to 1.",
        dest="device",
        type=int,
        default=1,
        required=False,
    )
    parser.add_argument(
        "--force",
        help="If you would like to start training without any questions",
        dest="force",
        action="store_true",
        required=False,
    )
    return parser


def add_train_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--config-file",
        help='path to config file (allowed are yaml, toml and json). Defaults to: "config/train.yaml"',
        dest="config_file",
        type=str,
        default="config/train.yaml",
        required=False,
    )
    parser.add_argument(
        "--device",
        help="set to a number to indicate multiple devices. Defaults to 1.",
        dest="device",
        type=int,
        default=1,
        required=False,
    )
    parser.add_argument(
        "--force",
        help="If you would like to start training without any questions",
        dest="force",
        action="store_true",
        required=False,
    )
    return parser


def add_generate_data_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--dataset-type",
        help="dataset_type for dataset: currently available: moon",
        dest="dataset_type",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--sizes",
        help="how many samples you want to create",
        dest="sizes",
        type=int,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--path",
        help="directory where you want to store the dataset",
        dest="path",
        type=str,
        default="./data",
        required=False,
    )
    return parser


def setup_process_parser(parser: ArgumentParser) -> ArgumentParser:
    command_subparser = parser.add_subparsers(dest="command", title="command")
    generate_data = command_subparser.add_parser(
        "generate-data",
        help="creates a specified dataset and stores it into the file system.",
    )
    generate_data = add_generate_data_args(generate_data)
    train = command_subparser.add_parser(
        "train", help="start training process as defined in your config file"
    )
    train = add_train_args(train)
    train_theta_noise = command_subparser.add_parser(
        "train-theta-noise",
        help="start training process as defined in your config file",
    )
    train_theta_noise = add_train_theta_noise_args(train_theta_noise)
    mcmc_sample = command_subparser.add_parser("mcmc-sample", help="sample mcmc")
    mcmc_sample = add_mcmc_sample_args(mcmc_sample)
    return parser


def setup_parser(parser: ArgumentParser) -> ArgumentParser:
    parser = setup_process_parser(parser)
    return parser
