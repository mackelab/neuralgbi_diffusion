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
            api.hydra_plugin.hydra_wrapper(
                module.train_potential,
                args,
                command_parser["train_potential"],
                config_var_name="config",
                version_base=None,
                config_path=str(Path.cwd().joinpath("config")),
                config_name="train_potential.yaml",
            )

        case "train-guidance":
            api.hydra_plugin.hydra_wrapper(
                module.train_guidance,
                args,
                command_parser["train_guidance"],
                config_var_name="config",
                version_base=None,
                config_path=str(Path.cwd().joinpath("config")),
                config_name="train_guidance.yaml",
            )

        case "train-diffusion":
            api.hydra_plugin.hydra_wrapper(
                module.train_diffusion,
                args,
                command_parser["train_diffusion"],
                config_var_name="config",
                version_base=None,
                config_path=str(Path.cwd().joinpath("config")),
                config_name="train_diffusion.yaml",
            )

        case "diffusion-sample":
            api.hydra_plugin.hydra_wrapper(
                module.diffusion_sample,
                args,
                command_parser["diffusion_sample"],
                config_var_name="config",
                version_base=None,
                config_name="sampling_diffusion.yaml",
                config_path=str(Path.cwd().joinpath("config/")),
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

        case "evaluate-diffusion-sampling":
            api.hydra_plugin.hydra_wrapper(
                module.evaluate_diffusion_sampling,
                args,
                command_parser["evaluate_diffusion_sampling"],
                config_var_name="config",
                version_base=None,
                config_name="evaluate_diffusion.yaml",
                config_path=str(Path.cwd().joinpath("config/")),
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
