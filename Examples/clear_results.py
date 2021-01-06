import subprocess
import sys
from pathlib import Path
import shutil


def clear():
    for path in Path(__file__).parent.rglob("results"):
        print(path)
        shutil.rmtree(path)

def main():
    "create simulations by directory"
    clear()

if __name__ == "__main__":
    main()