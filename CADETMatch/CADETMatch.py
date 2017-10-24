
import sys
import evo
import grad
import gradFD
import util
import time
import numpy

#due to how scoop works and the need to break things up into multiple processes it is hard to use class based systems
#As a result most of the code is broken up into modules but is still based on pure functions

def main():
    evo.createDirectories(evo.settings)
    evo.createCSV(evo.settings, evo.headers)
    evo.setupTemplates(evo.settings, evo.target)
    #grad.setupTemplates(evo.settings, evo.target)
    hof = evo.run(evo.settings, evo.toolbox)

    print("hall of fame")
    for i in hof:
        print(i, type(i), i.fitness.values)

    if "bootstrap" in evo.settings:
        temp = []

        samples = int(evo.settings['bootstrap']['samples'])
        center = float(evo.settings['bootstrap']['center'])
        noise = float(evo.settings['bootstrap']['percentNoise'])/100.0

        for i in range(samples):
            #copy csv files to a new directory with noise added
            #put a new json file in the directory that points to the new csv files
            json_path = util.copyCSVWithNoise(i, center, noise)
            print(json_path)

            #call setup on all processes with the new json file as an argument to reset them
            #util.updateScores(json_path)

            #hof = evo.run(evo.settings, evo.toolbox)
            #temp.extend([i.fitness.values for i in hof])

        #temp = numpy.array(temp)
        #print("cov", numpy.cov(temp), "data", temp)



if __name__ == "__main__":
    start = time.time()
    main()
    print("System has finished")
    print("The total runtime was %s seconds" % (time.time() - start))