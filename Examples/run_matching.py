import create_examples
import create_config
import subprocess
import sys
from pathlib import Path

def run_matching():
    for path in Path(__file__).parent.rglob("*.json"):
        if not (path.parent / "results").exists() and path.parent.name != "results":
            print(path)
            command = [sys.executable, '-m', 'CADETMatch', '--match', '-j', path.as_posix()]
            subprocess.run(command)

def main():
    "create simulations by directory"
    create_examples.main()
    create_config.main()
    run_matching()

if __name__ == "__main__":
    main()
