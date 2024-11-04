from argparse import ArgumentParser


def add_generate_data_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--dataset-type",
        help="dataset_type for dataset: currently available: moon",
        dest="dataset_type",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--size",
        help="how many samples you want to create",
        dest="size",
        type=int,
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
    parser.add_argument(
        "--noise-std",
        help="hwo to noise the data. Defaults to 0.01.",
        dest="noise_std",
        type=float,
        default=0.01,
        required=False,
    )
    return parser


def setup_experiment_parser(parser: ArgumentParser) -> ArgumentParser:
    command_subparser = parser.add_subparsers(dest="command", title="command")
    generate_data = command_subparser.add_parser(
        "generate-data",
        help="creates a specified dataset and stores it into the file system.",
    )
    generate_data = add_generate_data_args(generate_data)
    return parser


def setup_parser(parser: ArgumentParser) -> ArgumentParser:
    parser = setup_experiment_parser(parser)
    return parser
