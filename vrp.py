#!/usr/bin/env python3

#############################################
# GENETIC ALGORITHM FOR DVRP                #
# AUTHORS: Jakub Szymkowiak, Szymon Startek #
# SUBJECT: Optimization algorithms          #
#############################################


from argparse import ArgumentParser
import csv
from math import dist
from operator import ne
from platform import node
import random
import sys
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

def fit(vrp: GraphVrp, route):
  d = distance(vrp.depot, vrp.nodes[route[0]])
  for i in range(len(route) - 1):
    previous = vrp.nodes[route[i]]
    next = vrp.nodes[route[i + 1]]
    d += distance(previous, next)
  d += distance(vrp.nodes[route[len(route) - 1]], vrp.depot)
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


  return route

def genetic_algorithm(vrp: GraphVrp, iterations, popsize):
  # generage a random initial population
  population = []
  for i in range(popsize):
    p = range(1, len(vrp.nodes))
    random.shuffle(p)
    population.append(p)
  
  for p in population:
    adjust(p)
  
  for i in range(iterations):
    nextPopulation = []
    for j in range(len(population) / 2):
      parentIds = set()
      while len(parentIds) < 4:
        parentIds |= {random.randint(0, len(population) - 1)}
      parentIds = list(parentIds)

      if fit(population[parentIds[0]]) < fit(population[parentIds[1]]):
        parent1 = population[parentIds[0]]
      else:
        parent1 = population[parentIds[1]]

      if fit(population[parentIds[2]]) < fit(population[parentIds[3]]):
        parent2 = population[parentIds[2]]
      else:
        parent2 = population[parentIds[3]]

      # Crossover

      cutPoint1, cutPoint2 = random.randint(1, min(len(parent1), len(parent2)) - 1), random.randint(1, min(len(parent1), len(parent2)) - 1)
      cutPoint1, cutPoint2 = min(cutPoint1, cutPoint2), max(cutPoint1, cutPoint2)

      child1 = parent1[:cutPoint1] + parent2[cutPoint1:cutPoint2] + parent1[cutPoint2:]
      child2 = parent2[:cutPoint1] + parent1[cutPoint1:cutPoint2] + parent2[cutPoint2:]
      nextPopulation += [child1, child2]

      # Mutation

    if random.randint(1, 15) == 1:
      mutatedPopulation = nextPopulation[random.randint(0, len(nextPopulation) - 1)]
      i1 = random.randint(0, len(mutatedPopulation) - 1)
      i2 = random.randint(0, len(mutatedPopulation) - 1)
      mutatedPopulation[i1], mutatedPopulation[i2] = mutatedPopulation[i2], mutatedPopulation[i1]

    for r in nextPopulation:
      r = adjust(r)
    population = nextPopulation

  better = None
  bf = sys.float_info.max
  for r in population:
    f = fit(r)
    if f < bf:
      bf = f
      better = r
    
  return better




parser = ArgumentParser()
parser.add_argument("--input", "-i", help="input data")
parser.add_argument("--max", help="maximal dinstance for one truck")
parser.add_argument("--iterations", help="the number of generations of the genetic algorithm")
parser.add_argument("--popsize", help="population size")
args = parser.parse_args()

vrp = process_input_data(args.input, int(args.max))

genetic_algorithm(vrp, int(args.iterations), int(args.popsize))