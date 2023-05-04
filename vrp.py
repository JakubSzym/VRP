#!/usr/bin/env python3

#############################################
# GENETIC ALGORITHM FOR DVRP                #
# AUTHORS: Jakub Szymkowiak, Szymon Startek #
# SUBJECT: Optimization algorithms          #
#############################################


from argparse import ArgumentParser
import csv
from math import dist
from platform import node
import random
import numpy as np

class Node:
  def __init__(self, label,x, y) -> None:
    self.label = label
    self.x = x
    self.y = y
      
class GraphVrp:
  def __init__(self, maxdist, nodes) -> None:
    self.maxdist = maxdist
    self.nodes   = nodes
    self.depot   = Node(label="depot", x=0, y=0)

def process_input_data(filename, maxdist):
  nodes = []
  with open(filename, "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
      nodes.append(Node(row['label'], float(row['x']), float(row['y'])))
    
  return GraphVrp(maxdist, nodes)

def distance(node1, node2):
  return np.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

def fit(vrp: GraphVrp, population):
  d = distance(vrp.depot, vrp.nodes[population[0]])
  for i in range(len(population) - 1):
    previous = vrp.nodes[population[i]]
    next = vrp.nodes[population[i + 1]]
    d += distance(previous, next)
  d += distance(vrp.nodes[population[len(population) - 1]], vrp.depot)
  return d

def adjust(vrp: GraphVrp, route):
  repeated = True
  while repeated:
    repeated = False
    for i in range(len(route)):
      for j in range(i):
        if route[i] == route[j]:
          all = True
          for id in range(len(vrp.nodes)):
            if id not in route:
              route[i] = id
              all = False
              break
          if all:
            del route[i]
          repeated = True

        if repeated:
          break
      if repeated:
        break
  # adjust distance constraint???
  # To implement


  return population

def genetic_algorithm(vrp: GraphVrp, iterations, popsize):
  # generage a random initial population
  population = []
  for i in range(popsize):
    p = range(1, len(vrp.nodes))
    random.shuffle(p)
    population.append(p)
  
  for p in population:
    adjust(p)
  
  # for i in range(iterations):
  #   nextPopulation = []
  #   for j in range(len(population) / 2):
  #     parentIds = set()
  #     while len(parentIds) < 4:
  #       parentIds |= {random.randint(0, len(population) - 1)}
  #     parentIds = list(parentIds)
  return 0




parser = ArgumentParser()
parser.add_argument("--input", "-i", help="input data")
parser.add_argument("--max", help="maximal dinstance for one truck")
parser.add_argument("--iterations", help="the number of generations of the genetic algorithm")
parser.add_argument("--popsize", help="population size")
args = parser.parse_args()

vrp = process_input_data(args.input, int(args.max))

genetic_algorithm(vrp, int(args.iterations), int(args.popsize))