import subprocess
import sys
from pathlib import Path

def run_matching():
    for path in Path(sys.argv[1]).parent.rglob("*.json"):
        if not (path.parent / "results").exists() and path.parent.name != "results" and path.parent.name != "mcmc_refine":
            print(path)
            command = [sys.executable, '-m', 'CADETMatch', '--match', '-j', path.as_posix()]
            subprocess.run(command)

if __name__ == "__main__":
    run_matching()
