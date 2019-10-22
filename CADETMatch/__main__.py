import importlib
import argparse
import sys
import subprocess
import psutil

def makeParser():
    """Create the CADETMatch module arguments parser."""
    parser = argparse.ArgumentParser(
        description="Starts a parallel version of match using SCOOP.",
        prog="{0} -m CADETMatch.scoop_match".format(sys.executable),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--json', '-j',
                       help="Path to JSON file",
                       metavar="JSON")
    parser.add_argument('--match',
                        help="Run CADETMatch with Parameter Estimation and Error Modeling",
                        action='store_true')

    parser.add_argument('--generate_corner',
                        help="Generate corner graphs",
                        action='store_true')

    parser.add_argument('--generate_graphs',
                        help="Generate general graphs",
                        action='store_true')

    parser.add_argument('--generate_spearman',
                        help="Generate spearman graphs",
                        action='store_true')
    
    parser.add_argument('--generate_mcmc_plot_tube',
                        help="Generate mcmc plot tube graphs",
                        action='store_true')

    parser.add_argument('--generate_mle',
                        help="Generate maximum liklihood",
                        action='store_true')

    parser.add_argument('-n',
                        help="Number of scoop processes to use. Use 1 for debugging",
                        default=0)

    return parser


def run_command(module, json, number_of_jobs, additional=None):
    command = [sys.executable, '-m', 'scoop']
    if int(number_of_jobs) > 0:
        command.extend(['-n', str(number_of_jobs)])
    #else:
    #    ncpus = psutil.cpu_count(logical=False)
    #    command.extend(['-n', str(ncpus)])

    command.extend([importlib.util.find_spec(module).origin, str(json)])
    if additional is not None:
        command.extend(additional)

    rc = subprocess.run(command, bufsize=1)
    
    return rc.returncode


if __name__ == "__main__":
    parser = makeParser()
    args = parser.parse_args()
    
    if args.match:
        sys.exit(run_command('CADETMatch.match', args.json, args.n))
    if args.generate_corner:
        sys.exit(run_command('CADETMatch.generate_corner_graphs', args.json, args.n))
    if args.generate_graphs:
        sys.exit(run_command('CADETMatch.generate_graphs', args.json, args.n, ['2']))
    if args.generate_spearman:
        sys.exit(run_command('CADETMatch.graph_spearman', args.json, args.n, ['end']))
    if args.generate_mcmc_plot_tube:
        sys.exit(run_command('CADETMatch.mcmc_plot_tube', args.json, args.n))
    if args.generate_mle:
        sys.exit(run_command('CADETMatch.mle', args.json, args.n))
