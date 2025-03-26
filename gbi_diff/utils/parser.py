from argparse import ArgumentParser
from typing import Tuple, Dict, List


def add_mcmc_sample_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--checkpoint",
        help="path to checkpoint",
        dest="checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--n-samples",
        help="how many samples you would like to sample. Defaults to 100.",
        dest="n_samples",
        type=int,
        default=100,
        required=False,
    )
    parser.add_argument(
        "--config-file",
        help='path to config file. Defaults to "config/mcmc.yaml".',
        dest="config_file",
        type=str,
        default="config/sampling_mcmc.yaml",
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


def add_diffusion_sample_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--diffusion-ckpt",
        help="path to checkpoint of guidance model",
        dest="diffusion_ckpt",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--guidance-ckpt",
        help='path to config to use for diffusion sample. Defaults to "config/sampling_diffusion.yaml".',
        dest="guidance_ckpt",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output",
        help="Where to store the samples. If None: store alongside diffusion process. Defaults to None.",
        dest="output",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--n-samples",
        help="How many samples to sample from posterior distribution. Defaults to 100.",
        dest="n_samples",
        type=int,
        default=100,
        required=False,
    )
    parser.add_argument(
        "--plot",
        help="Would you like to add pair plots of the posterior distribution. Defaults to False.",
        dest="plot",
        action="store_true",
        required=False,
    )
    return parser


def add_train_diffusion_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--device",
        help="_description_. Defaults to 1.",
        dest="device",
        type=int,
        default=1,
        required=False,
    )
    parser.add_argument(
        "--force",
        help="_description_. Defaults to False.",
        dest="force",
        action="store_true",
        required=False,
    )
    return parser


def add_train_guidance_args(parser: ArgumentParser) -> ArgumentParser:
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


from pyargwriter.api.hydra_plugin import add_hydra_parser


def add_train_potential_args(parser: ArgumentParser) -> ArgumentParser:
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


def setup_entrypoint_parser(
    parser: ArgumentParser,
) -> Tuple[ArgumentParser, Dict[str, ArgumentParser]]:
    subparser = {}
    command_subparser = parser.add_subparsers(dest="command", title="command")
    generate_data = command_subparser.add_parser(
        "generate-data",
        help="creates a specified dataset and stores it into the file system.",
    )
    generate_data = add_generate_data_args(generate_data)
    subparser["generate_data"] = generate_data
    train_potential = command_subparser.add_parser(
        "train-potential", help="start training process as defined in your config file"
    )
    train_potential = add_train_potential_args(train_potential)
    train_potential = add_hydra_parser(train_potential)
    subparser["train_potential"] = train_potential
    train_guidance = command_subparser.add_parser(
        "train-guidance", help="start training process as defined in your config file"
    )
    train_guidance = add_train_guidance_args(train_guidance)
    train_guidance = add_hydra_parser(train_guidance)
    subparser["train_guidance"] = train_guidance
    train_diffusion = command_subparser.add_parser(
        "train-diffusion",
        help="train diffusion model which is also the prior for the sampling process",
    )
    train_diffusion = add_train_diffusion_args(train_diffusion)
    train_diffusion = add_hydra_parser(train_diffusion)
    subparser["train_diffusion"] = train_diffusion
    diffusion_sample = command_subparser.add_parser(
        "diffusion-sample", help="sample from diffusion process"
    )
    diffusion_sample = add_diffusion_sample_args(diffusion_sample)
    diffusion_sample = add_hydra_parser(diffusion_sample)
    subparser["diffusion_sample"] = diffusion_sample
    mcmc_sample = command_subparser.add_parser("mcmc-sample", help="sample mcmc")
    mcmc_sample = add_mcmc_sample_args(mcmc_sample)
    subparser["mcmc_sample"] = mcmc_sample
    return parser, subparser


def setup_parser(parser: ArgumentParser) -> ArgumentParser:
    parser, _ = setup_entrypoint_parser(parser)
    return parser
