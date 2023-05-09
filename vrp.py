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
from PIL import Image, ImageDraw

class Node:
  def __init__(self, label,x, y) -> None:
    self.label = label
    self.x = x
    self.y = y
      
class GraphVrp:
  def __init__(self, maxdist, nodes, trucks) -> None:
    self.maxdist = maxdist
    self.trucks  = trucks
    self.nodes   = nodes

def process_input_data(filename, maxdist, trucks):
  nodes = []
  with open(filename, "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
      nodes.append(Node(row['label'], float(row['x']), float(row['y'])))
    
  return GraphVrp(maxdist, nodes, trucks)

def distance(node1, node2):
  return np.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

def fit(vrp: GraphVrp, route):
  if route.count(0) > vrp.trucks+1:
    return 999999999

  d = 0#distance(vrp.depot, vrp.nodes[route[0]])
  for i in range(len(route) - 1):
    previous = vrp.nodes[route[i]]
    next = vrp.nodes[route[i + 1]]
    d += distance(previous, next)
  return d

def remove_zeros(route):
  i = len(route)-2
  while i >= 0:
    if route[i] == route[i+1] == 0:
      del route[i]
    i -= 1
  return route

def adjust(vrp: GraphVrp, route):
  route = remove_zeros(route)
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
  i = 0
  d = 0.0
  cap = vrp.maxdist
  #print(f"{route=}")
  #print(f"{len(route)=} and {len(vrp.nodes)=}")
  route.append(0)

  while i < len(route)-1:
    d += distance(vrp.nodes[route[i]], vrp.nodes[route[i+1]])
    #print(f"{d=}")
    if d > cap:
      route.insert(i, 0)
      d = 0.0
      #print(route)
      #break
    i += 1
  #i = len(route) - 2
  # To implement
  route = remove_zeros(route)
  return route

def genetic_algorithm(vrp: GraphVrp, iterations, popsize):
  # generage a random initial population
  population = []
  for i in range(popsize):
    p = [i for i in range(1, len(vrp.nodes))] #p = range(1, len(vrp.nodes))
    random.shuffle(p)
    p.insert(0, 0)

    population.append(p)
  
  for p in population:
    adjust(vrp, p)

  for i in range(iterations):
    nextPopulation = []
    for j in range(int(len(population) / 2)):
      parentIds = set()
      while len(parentIds) < 4:
        parentIds |= {random.randint(0, len(population) - 1)}
      parentIds = list(parentIds)

      if fit(vrp,population[parentIds[0]]) < fit(vrp,population[parentIds[1]]):
        parent1 = population[parentIds[0]]
      else:
        parent1 = population[parentIds[1]]

      if fit(vrp,population[parentIds[2]]) < fit(vrp,population[parentIds[3]]):
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
      r = adjust(vrp, r)
    population = nextPopulation

  better = None
  bf = sys.float_info.max
  #print(f"{population=}")
  for r in population:
    f = fit(vrp, r)
    #print(f"{f=}")
    if f < bf:
      bf = f
      better = r
  print(f"{bf=}")
  return better

def draw(vrp: GraphVrp, route):
  w, h = 2000, 2000

  img = Image.new("RGB", (w, h), "white")
  draw = ImageDraw.Draw(img)

  for i, point in enumerate(vrp.nodes):
    x, y = point.x*100+w/2, point.y*100+h/2
    color = "blue" if i > 0 else "red"
    draw.ellipse((x-10,y-10, x+10, y+10), fill=color, outline=(0, 0, 0))

    draw.text((x,y), str(point.label), fill="black")

  for i in range(len(route)-1):
    begin = vrp.nodes[route[i]]
    end = vrp.nodes[route[i+1]]
    draw.line([(begin.x*100+w/2, begin.y*100+h/2), (end.x*100+w/2, end.y*100+h/2)], fill="black")

  img.show()

parser = ArgumentParser()
parser.add_argument("--input", "-i", help="input data")
parser.add_argument("--max", help="maximal dinstance for one truck")
parser.add_argument("--trucks", help="maximal number of truck")
parser.add_argument("--iterations", help="the number of generations of the genetic algorithm")
parser.add_argument("--popsize", help="population size")
args = parser.parse_args()

vrp = process_input_data(args.input, int(args.max), int(args.trucks))

best = genetic_algorithm(vrp, int(args.iterations), int(args.popsize))
print(best)

draw(vrp, best)