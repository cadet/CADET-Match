
import sys
import evo
import util

#due to how scoop works and the need to break things up into multiple processes it is hard to use class based systems
#As a result most of the code is broken up into modules but is still based on pure functions

def main():

    evo.createCSV(evo.settings, evo.headers)
    evo.setupTemplates(evo.settings, evo.target)
    pop, logbook, hof = evo.run(evo.settings, evo.toolbox)

    return pop, logbook, hof


if __name__ == "__main__":
    main()
    print("System has finished")