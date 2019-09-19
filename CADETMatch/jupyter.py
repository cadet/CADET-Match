import CADETMatch.match
import IPython.display as dp
from CADETMatch.cache import Cache
import subprocess
import sys
import pandas

class Match:
    def __init__(self, json_path):
        self.json_path = json_path
        self.cache = Cache()
        self.cache.setup(json_path)

    def start_sim(self):
        command = [sys.executable, '-m', 'scoop', CADETMatch.match.__file__, str(self.json_path), str(1)]
        pipe = subprocess.PIPE

        proc = subprocess.Popen(command, stdout=pipe, stderr=subprocess.STDOUT, bufsize=1)
    
        for line in iter(proc.stdout.readline, b''):
            print(line.decode('utf-8'))
        return proc.poll()

    def plot_corner(self):
        process_dir = self.cache.settings['resultsDirProgress']

        corner = process_dir / 'corner.png'
        corner_transform = process_dir / 'corner_transform.png'

        if corner.exists():
            print("Corner plot in search space")
            image = dp.Image(filename=corner, embed=True)
            dp.display(image)

        if corner_transform.exists():
            print("Corner plot in original space")
            image = dp.Image(filename=corner_transform, embed=True)
            dp.display(image)

    def plot_best(self):
        meta_dir = self.cache.settings['resultsDirMeta']
        csv = meta_dir / 'results.csv'
        data = pandas.read_csv(csv.as_posix())
    
        best = {}
        score = {}
    
        cols = [('Product Root Score', False), ('Min Score', False), ('Mean Score', False), ('SSE', False)]
    
        for col_name, order in cols:
            temp = data.sort_values(by=[col_name,], ascending=order)
            head = temp.head(1)
            name = str(head.Name.iloc[0])
            score[name] = head
            if name not in best:
                best[name] = [col_name]
            else:
                best[name].append(col_name)
            
        for name, col_names in best.items():
        
            print("Best item %s for meta score(s) %s" % (name, ' , '.join(col_names)))
            dp.display(score[name])
            images = meta_dir.glob('%s_*.png' % name)
            for image in images:
                img = dp.Image(filename=image, embed=True)
                dp.display(img)