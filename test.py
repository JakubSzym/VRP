import csv
from random import shuffle, randint
import sys
import numpy as np
from PIL import Image, ImageDraw

parent1 = [1,2,3,4,5,6,7,8,9]
parent2 = [9,8,7,6,5,4,3,2,1]

cut_point1 = randint(0, min(len(parent1), len(parent2)))
cut_point2 = randint(0, min(len(parent1), len(parent2)))
cut_point1, cut_point2 = min(cut_point1, cut_point2), max(cut_point1, cut_point2)
print(f"{cut_point1=}, {cut_point2=}")

cut1 = parent1[cut_point1:cut_point2]
cut2 = parent2[cut_point1:cut_point2]

par_list1 = [ele for ele in parent1 if not ele in cut2]
par_list2 = [ele for ele in parent2 if not ele in cut1]

child1 = par_list1[:cut_point1] + cut2 + par_list1[cut_point1:]
child2 = par_list2[:cut_point1] + cut1 + par_list2[cut_point1:]

next_population = [child1, child2]

print(next_population)