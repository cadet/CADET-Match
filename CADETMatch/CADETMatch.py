
import sys
import evo
import grad
import gradFD
import util
import time
import faulthandler

#due to how scoop works and the need to break things up into multiple processes it is hard to use class based systems
#As a result most of the code is broken up into modules but is still based on pure functions

def main():
    evo.createDirectories(evo.settings)
    evo.createCSV(evo.settings, evo.headers)
    evo.setupTemplates(evo.settings, evo.target)
    #grad.setupTemplates(evo.settings, evo.target)
    evo.run(evo.settings, evo.toolbox)


if __name__ == "__main__":
    f = open('cadet_trace', 'w')
    faulthandler.dump_traceback_later(120, repeat=True, file=f, exit=False)
    start = time.time()
    main()
    print("System has finished")
    print("The total runtime was %s seconds" % (time.time() - start))