from argparse import ArgumentParser
from gbi_diff.entrypoint import Process
from gbi_diff.utils.parser import setup_parser


def execute(args: dict) -> bool:
    module = Process()
    match args["command"]:
        case "generate-data":
            module.generate_data(
                dataset_type=args["dataset_type"],
                sizes=args["sizes"],
                path=args["path"],
            )

        case "train":
            module.train(
                config_file=args["config_file"],
                device=args["device"],
                force=args["force"],
            )

        case "mcmc-sample":
            module.mcmc_sample(
                checkpoint=args["checkpoint"],
                observed_data=args["observed_data"],
                size=args["size"],
                config_file=args["config_file"],
                output=args["output"],
                plot=args["plot"],
            )

        case _:
            return False

    return True


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(description="CLI Process to handle GBI pipeline")

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
