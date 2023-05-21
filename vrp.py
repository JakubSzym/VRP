#!/usr/bin/env python3

#############################################
# GENETIC ALGORITHM FOR DVRP                #
# AUTHORS: Jakub Szymkowiak, Szymon Startek #
# SUBJECT: Optimization algorithms          #
#############################################


from argparse import ArgumentParser
import functions as fun

parser = ArgumentParser()
parser.add_argument("--input", "-i", help="input data")
parser.add_argument("--max", help="maximal distance for one truck")
parser.add_argument("--trucks", help="maximal number of truck")
parser.add_argument("--iterations", help="the number of generations of the genetic algorithm")
parser.add_argument("--popsize", help="population size")
args = parser.parse_args()

dvrp = fun.process_input_data(args.input, int(args.max), int(args.trucks))
best = fun.genetic_algorithm(dvrp, int(args.iterations), int(args.popsize))
print(best)
fun.draw(dvrp, best[0])
