import csv
from random import shuffle, randint
import sys
import numpy as np
from PIL import Image, ImageDraw

population = [10, 9, 1, 4, 8, 7, 7, 2, 1]
pop = []

for j in range(int(len(population) / 2)):
    parent_ids = set()
    while len(parent_ids) < 4:
        parent_ids |= {randint(0, len(population) - 1)}
    parent_ids = list(parent_ids)
    print(f"{parent_ids=}")

    if population[parent_ids[0]] < population[parent_ids[1]]:
        parent1 = population[parent_ids[0]]
    else:
        parent1 = population[parent_ids[1]]

    if population[parent_ids[2]] < population[parent_ids[3]]:
        parent2 = population[parent_ids[2]]
    else:
        parent2 = population[parent_ids[3]]

    pop += [parent1, parent2]
    print(f"{parent1=}, {parent2=}")
print(f"{pop=}")


###
print(f"")

population = [10, 9, 1, 4, 8, 7, 7, 2, 1]
pop = []

population.sort(reverse=True, key=lambda item: item)
print(f"{population=}")

population_len = len(population)
for j in range(int(len(population) / 2)):
    if j % 2 == 0:
        parent1 = population[j]
        parent2 = population[j+1]
    else:
        parent1 = population[j]
        parent2 = population[j+randint(1, int(population_len/2))]

        pop += [parent1, parent2]
        print(f"{parent1=}, {parent2=}")
print(f"{pop=}")