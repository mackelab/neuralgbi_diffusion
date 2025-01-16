from argparse import ArgumentParser
from pathlib import Path
from pyargwriter import api
from gbi_diff.utils.parser import setup_entrypoint_parser
from gbi_diff.entrypoint import Entrypoint
from gbi_diff.utils.parser import setup_parser


def execute(args: dict) -> bool:
    module = Entrypoint()
    _, command_parser = setup_entrypoint_parser(ArgumentParser())
    match args["command"]:
        case "generate-data":
            module.generate_data(
                dataset_type=args["dataset_type"],
                sizes=args["sizes"],
                path=args["path"],
            )

        case "train-potential":
            module.train_potential(
                config_file=args["config_file"],
                device=args["device"],
                force=args["force"],
            )

        case "train-guidance":
            module.train_guidance(
                config_file=args["config_file"],
                device=args["device"],
                force=args["force"],
            )

        case "train-diffusion":
            module.train_diffusion(
                config_file=args["config_file"],
                device=args["device"],
                force=args["force"],
            )

        case "diffusion-sample":
            module.diffusion_sample(
                diffusion_ckpt=args["diffusion_ckpt"],
                guidance_ckpt=args["guidance_ckpt"],
                config=args["config"],
                output=args["output"],
                n_samples=args["n_samples"],
                plot=args["plot"],
            )

        case "mcmc-sample":
            module.mcmc_sample(
                checkpoint=args["checkpoint"],
                n_samples=args["n_samples"],
                config_file=args["config_file"],
                output=args["output"],
                plot=args["plot"],
                num_worker=args["num_worker"],
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
