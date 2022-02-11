import CADETMatch.match
import CADETMatch.util as util
import sys

print(sys.argv)

sys.argv.extend(["F:/cadet_release_test/search/unsga3/dextran.json", "1"])

if __name__ == '__main__':
    map_function = util.getMapFunction()
    print(map_function)
    CADETMatch.match.main(map_function=map_function)