#!/usr/bin/env python3

import random
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--filename", help="output file")
parser.add_argument("--nodes",    help="number of nodes")
parser.add_argument("--maxdist",  help="maximal distance for one truck")
parser.add_argument("--minX",     help="minimal value of x coordinate")
parser.add_argument("--maxX",     help="maximal value of x coordinate")
parser.add_argument("--minY",     help="minimal value of y coordinate")
parser.add_argument("--maxY",     help="maximal value of y coordinate")

args = parser.parse_args()

filename = args.filename
nodes    = int(args.nodes)
maxdist  = int(args.maxdist)
minX     = int(args.minX)
maxX     = int(args.maxX)
minY     = int(args.minY)
maxY     = int(args.maxY)

with open(filename, "w") as file:
  file.write("params:\n")
  file.write(f"maxdist {maxdist}\n")
  file.write("nodes:\n")
  for i in range(nodes):
    demand = random.randint(0, maxdist)
    x = random.uniform(minX, maxX)
    y = random.uniform(minY, maxY)
    file.write(f" N{i+1} {demand} {x} {y}\n")