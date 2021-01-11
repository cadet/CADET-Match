import argparse
import importlib
import subprocess
import sys
import pathlib

import psutil


def makeParser():
    """Create the CADETMatch module arguments parser."""
    parser = argparse.ArgumentParser(
        description="Starts a parallel version of match using multiprocessing.",
        prog="{0} -m CADETMatch".format(sys.executable),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--json", "-j", help="Path to JSON file", metavar="JSON")
    parser.add_argument(
        "--match",
        help="Run CADETMatch with Parameter Estimation and Error Modeling",
        action="store_true",
    )

    parser.add_argument(
        "--generate_corner", help="Generate corner graphs", action="store_true"
    )

    parser.add_argument(
        "--generate_graphs",
        help="Generate general graphs excluding 3D",
        action="store_true",
    )

    parser.add_argument(
        "--generate_graphs_all",
        help="Generate general graphs including 3D",
        action="store_true",
    )

    parser.add_argument(
        "--generate_graphs_autocorr",
        help="Generate autocorrelation graphs",
        action="store_true",
    )

    parser.add_argument(
        "--generate_spearman", help="Generate spearman graphs", action="store_true"
    )

    parser.add_argument(
        "--generate_mcmc_plot_tube",
        help="Generate mcmc plot tube graphs",
        action="store_true",
    )

    parser.add_argument(
        "--generate_mcmc_mixing",
        help="Generate mcmc mixing graphs",
        action="store_true",
    )

    parser.add_argument(
        "--generate_mle", help="Generate maximum liklihood", action="store_true"
    )

    parser.add_argument(
        "-n", help="Number of parallel processes to use. Use 1 for debugging", default=0
    )

    parser.add_argument(
        "--generate_examples", help="Directory for CADETMatch Example generation", action="store", 
        type=pathlib.Path
    )

    parser.add_argument(
        "--cadet_examples", help="Set path to the cadet-cli binary for examples to use", action="store", 
        type=pathlib.Path
    )

    parser.add_argument(
        "--run_examples", help="Directory for CADETMatch Example running", action="store", 
        type=pathlib.Path
    )

    parser.add_argument(
        "--clean_examples", help="Directory for CADETMatch Example cleaning (remove results directories)", action="store_true", 
        type=pathlib.Path
    )    

    return parser


def run_command(module, json, number_of_jobs, additional=None):
    command = [
        sys.executable,
    ]
    command.extend([importlib.util.find_spec(module).origin, str(json)])
    if additional is not None:
        command.extend(additional)

    command.append(str(number_of_jobs))

    rc = subprocess.run(command, bufsize=1)

    return rc.returncode


if __name__ == "__main__":
    parser = makeParser()
    args = parser.parse_args()

    if args.generate_examples and args.cadet_examples is None:
        parser.error("--generate_examples requires --cadet_examples")

    if args.match:
        sys.exit(run_command("CADETMatch.match", args.json, args.n))
    if args.generate_corner:
        sys.exit(run_command("CADETMatch.generate_corner_graphs", args.json, args.n))
    if args.generate_graphs_autocorr:
        sys.exit(run_command("CADETMatch.generate_autocorr_graphs", args.json, args.n))
    if args.generate_mcmc_mixing:
        sys.exit(run_command("CADETMatch.generate_mixing_graphs", args.json, args.n))
    if args.generate_graphs:
        sys.exit(run_command("CADETMatch.generate_graphs", args.json, args.n, ["1"]))
    if args.generate_graphs_all:
        sys.exit(run_command("CADETMatch.generate_graphs", args.json, args.n, ["2"]))
    if args.generate_spearman:
        sys.exit(run_command("CADETMatch.graph_spearman", args.json, args.n, ["end"]))
    if args.generate_mcmc_plot_tube:
        sys.exit(run_command("CADETMatch.mcmc_plot_tube", args.json, args.n))
    if args.generate_mle:
        sys.exit(run_command("CADETMatch.mle", args.json, args.n))
    if args.clean_examples:
        sys.exit(run_command("CADETMatch.Examples.clean_examples", args.clean_examples))
    if args.run_examples:
        sys.exit(run_command("CADETMatch.Examples.run_examples", args.run_examples))
    if args.generate_examples:
        sys.exit(run_command("CADETMatch.Examples.generate_examples", args.generate_examples, args.cadet_examples))
