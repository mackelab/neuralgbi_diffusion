from argparse import ArgumentParser
from gbi_diff.experiment import Experiment
from gbi_diff.utils.parser import setup_parser


def execute(args: dict) -> bool:
    module = Experiment()
    match args["command"]:
        case "generate-data":
            module.generate_data(
                dataset_type=args["dataset_type"],
                size=args["size"],
                path=args["path"],
                noise_std=args["noise_std"],
            )

        case _:
            return False

    return True


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(description="")

    parser = setup_parser(parser)

    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    args_dict = vars(args)
    if not execute(args_dict):
        parser.print_usage()


if __name__ == "__main__":
    main()
