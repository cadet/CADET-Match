import create_examples
import create_config
import subprocess
import sys
from pathlib import Path
from cadet import H5, Cadet
from addict import Dict

defaults = Dict()
defaults.cadet_path = Path(r"C:\Users\kosh_000\cadet_build\CADET\CADET41\bin\cadet-cli.exe").as_posix()
defaults.base_dir = Path(__file__).parent
defaults.flow_rate = 2.88e-8 # m^3/s
defaults.ncol = 100
defaults.npar = 10
defaults.abstol = 1e-8
defaults.algtol = 1e-10
defaults.reltol = 1e-8
defaults.lin_ka1 = 4e-4
defaults.lin_ka2 = 1e-4
defaults.lin_kd1 = 4e-3
defaults.lin_kd2 = 1e-3
defaults.col_dispersion = 2e-7
defaults.film_diffusion = 1e-6
defaults.par_diffusion = 3e-11
defaults.population = 20

Cadet.cadet_path = defaults.cadet_path

def run_matching():
    for path in Path(__file__).parent.rglob("*.json"):
        if not (path.parent / "results").exists() and path.parent.name != "results" and path.parent.name != "mcmc_refine":
            print(path)
            command = [sys.executable, '-m', 'CADETMatch', '--match', '-j', path.as_posix()]
            subprocess.run(command)

def main():
    "create simulations by directory"
    #create_examples.main(defaults)
    #create_config.main(defaults)
    run_matching()

if __name__ == "__main__":
    main()
