import CADETMatch.match
import IPython.display as dp
from CADETMatch.cache import Cache
import subprocess
import sys
import pandas
import psutil
import signal


class Match:
    def __init__(self, json_path):
        self.json_path = json_path
        self.cache = Cache()
        self.cache.setup_dir(json_path)
        CADETMatch.match.createDirectories(self.cache, json_path)
        self.cache.setup(json_path)

    def start_sim(self):
        # ncpus = psutil.cpu_count(logical=False)
        command = [sys.executable, CADETMatch.match.__file__, str(self.json_path), str(1)]  #'-n', str(ncpus),
        pipe = subprocess.PIPE

        proc = subprocess.Popen(command, stdout=pipe, stderr=subprocess.STDOUT, bufsize=1)

        def signal_handler(sig, frame):
            for child in psutil.Process(proc.pid).children(recursive=True):
                child.kill()
            proc.kill()
            print("Terminating")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        for line in iter(proc.stdout.readline, b""):
            print(line.decode("utf-8"))
        return proc.poll()

    def plot_corner(self):
        process_dir = self.cache.settings["resultsDirProgress"]

        corner = process_dir / "sns_corner.png"
        corner_transform = process_dir / "sns_corner_transform.png"

        if corner.exists():
            print("Corner plot in search space")
            image = dp.Image(filename=corner.as_posix(), embed=True)
            dp.display(image)

        if corner_transform.exists():
            print("Corner plot in original space")
            image = dp.Image(filename=corner_transform.as_posix(), embed=True)
            dp.display(image)

    def plot_best(self):
        meta_dir = self.cache.settings["resultsDirMeta"]
        best, score, best_score = self.get_best()

        for name, col_names in best.items():

            print("Best item %s for meta score(s) %s" % (name, " , ".join(col_names)))
            dp.display(score[name])
            images = meta_dir.glob("%s_*.png" % name)
            for image in images:
                img = dp.Image(filename=image.as_posix(), embed=True)
                dp.display(img)

    def get_best(self):
        "return the best values for each score"
        meta_dir = self.cache.settings["resultsDirMeta"]
        csv = meta_dir / "results.csv"
        data = pandas.read_csv(csv.as_posix())

        best = {}
        score = {}
        best_score = {}

        cols = [("Product Root Score", False), ("Min Score", False), ("Mean Score", False), ("SSE", False)]

        for col_name, order in cols:
            temp = data.sort_values(by=[col_name,], ascending=order)
            head = temp.head(1)
            name = str(head.Name.iloc[0])
            score[name] = head
            best_score[col_name] = head[self.cache.parameter_headers].values[0]
            if name not in best:
                best[name] = [col_name]
            else:
                best[name].append(col_name)

        return best, score, best_score

    def plot_space(self):
        space_dir = self.cache.settings["resultsDirSpace"] / "2d"

        for image in space_dir.glob("*.png"):
            if "1- " in image.name or "SSE" in image.name:
                print(image.name)
                img = dp.Image(filename=image.as_posix(), embed=True)
                dp.display(img)
