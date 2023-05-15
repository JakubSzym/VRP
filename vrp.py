#!/usr/bin/env python3

#############################################
# GENETIC ALGORITHM FOR DVRP                #
# AUTHORS: Jakub Szymkowiak, Szymon Startek #
# SUBJECT: Optimization algorithms          #
#############################################


from argparse import ArgumentParser
import csv
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
  d = max(0, route.count(0) - (vrp.trucks+1)) * 999999999
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
  route.insert(0, 0)
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
  return route

def add_zeros(vrp: GraphVrp, route):
  i = 0
  d = 0.0
  cap = vrp.maxdist
  route.insert(0, 0)
  route.append(0)

  while i < len(route) - 1:
    d += distance(vrp.nodes[route[i]], vrp.nodes[route[i + 1]])
    if d > cap:
      route.insert(i, 0)
      d = 0.0
    i += 1
  route = remove_zeros(route)
  return route

def genetic_algorithm(vrp: GraphVrp, iterations, popsize):
  # generage a random initial population
  population = []
  for i in range(popsize):
    p = [i for i in range(1, len(vrp.nodes))]
    random.shuffle(p)
    population.append(p)

  for i in range(iterations):
    zero_population = []
    for p in population:
      zero_population.append(add_zeros(vrp, p.copy()))

    fit_population = [fit(vrp, zero) for zero in zero_population]

    next_population = []
    for j in range(int(len(population) / 2)):
      parent_ids = set()
      while len(parent_ids) < 4:
        parent_ids |= {random.randint(0, len(population) - 1)}
      parent_ids = list(parent_ids)

      if fit_population[parent_ids[0]] < fit_population[parent_ids[1]]:
        parent1 = population[parent_ids[0]]
      else:
        parent1 = population[parent_ids[1]]

      if fit_population[parent_ids[2]] < fit_population[parent_ids[3]]:
        parent2 = population[parent_ids[2]]
      else:
        parent2 = population[parent_ids[3]]

      # Crossover

      cut_point1, cut_point2 = random.randint(1, min(len(parent1), len(parent2)) - 1), random.randint(1, min(len(parent1), len(parent2)) - 1)
      cut_point1, cut_point2 = min(cut_point1, cut_point2), max(cut_point1, cut_point2)

      child1 = parent1[:cut_point1] + parent2[cut_point1:cut_point2] + parent1[cut_point2:]
      child2 = parent2[:cut_point1] + parent1[cut_point1:cut_point2] + parent2[cut_point2:]
      next_population += [child1, child2]

      # Mutation

    #0 - 1% of population will mutate
    mutation_count = random.randint(0, popsize * 0.01)
    for j in range(mutation_count):
      mutated_population = next_population[random.randint(0, len(next_population) - 1)]
      i1 = random.randint(0, len(mutated_population) - 1)
      i2 = random.randint(0, len(mutated_population) - 1)
      mutated_population[i1], mutated_population[i2] = mutated_population[i2], mutated_population[i1]

    for j in range(len(next_population)):
      next_population[j] = adjust(vrp, next_population[j])
    population = next_population

  better = None
  bf = sys.float_info.max
  #print(f"{population=}")
  for r in population:
    f = fit(vrp, add_zeros(vrp, r))
    #print(f"{f=}")
    if f < bf:
      bf = f
      better = r
  print(f"{bf=}")
  return better

def draw(vrp: GraphVrp, route):
  w, h = 2000, 2000

  img = Image.new("RGB", (w, h), "white")
  draw_img = ImageDraw.Draw(img)

  for i, point in enumerate(vrp.nodes):
    x, y = point.x*100+w/2, point.y*100+h/2
    color = "blue" if i > 0 else "red"
    draw_img.ellipse((x-10,y-10, x+10, y+10), fill=color, outline=(0, 0, 0))

    draw_img.text((x,y), str(point.label), fill="black")

  for i in range(len(route)-1):
    begin = vrp.nodes[route[i]]
    end = vrp.nodes[route[i+1]]
    draw_img.line([(begin.x*100+w/2, begin.y*100+h/2), (end.x*100+w/2, end.y*100+h/2)], fill="black")

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

route = adjust(vrp, [9, 6, 1, 3, 2, 5, 7, 4, 8])
my = fit(vrp, add_zeros(vrp, route))
print(f"My try ({route}): {my}")
