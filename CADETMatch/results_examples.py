import subprocess
import sys
from pathlib import Path
import pandas

def run_matching():
    results = []
    baseDir = Path(sys.argv[1]).resolve()
    for path in sorted(baseDir.rglob("results.xlsx")):
        print(path)
        df = pandas.read_excel(path)
        results.append(path.as_posix())
        results.append(df.to_string(index=False))

    with (baseDir / "results.txt").open("w", newline="") as outfile:
        outfile.write('\n\n'.join(results))

if __name__ == "__main__":
    run_matching()

