import csv
from random import shuffle, randint
import sys
import numpy as np
from PIL import Image, ImageDraw

class Node:
    def __init__(self, label, x, y) -> None:
        self.label = label
        self.x = x
        self.y = y


class GraphDVRP:
    def __init__(self, maxdist, nodes, trucks) -> None:
        self.maxdist = maxdist
        self.trucks = trucks
        self.nodes = nodes


def process_input_data(filename, maxdist, trucks):
    nodes = []
    with open(filename, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            nodes.append(Node(row['label'], float(row['x']), float(row['y'])))

    return GraphDVRP(maxdist, nodes, trucks)


def distance(node1, node2):
    return np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)


def check_nodes_distance_from_depot(dvrp: GraphDVRP, PRINT = False):
    for node in dvrp.nodes:
        val = distance(node, dvrp.nodes[0])
        assert 2 * val <= dvrp.maxdist, f"Max distance is too small - at least {2*val}"
        if PRINT:
            print(val)


def check_best(dvrp: GraphDVRP, best):
    dist = 0
    for i,ele in enumerate(best[0]):
        if ele == 0 and i > 0:
            if dist > dvrp.maxdist:
                print("Check distance")
            dist = 0
        if i < len(best[0]) - 1:
            dist += distance(dvrp.nodes[best[0][i]], dvrp.nodes[best[0][i+1]])


def fit(dvrp: GraphDVRP, route):
    d = max(0, route.count(0) - (dvrp.trucks + 1)) * 100000000
    for i in range(len(route) - 1):
        previous = dvrp.nodes[route[i]]
        next = dvrp.nodes[route[i + 1]]
        d += distance(previous, next)
    return d


def remove_zeros(route):
    i = len(route) - 2
    while i >= 0:
        if route[i] == route[i + 1] == 0:
            del route[i]
        i -= 1
    return route


def add_zeros(dvrp: GraphDVRP, route):
    i = 0
    d = 0.0
    r = route.copy()
    r.insert(0, 0)
    r.append(0)
    #print(f"")
    while i < len(r) - 1:
        #print(f"{i=}, pre{r=}")
        #print(f"{d + distance(dvrp.nodes[r[i]], dvrp.nodes[r[0]])}")
        if d + distance(dvrp.nodes[r[i]], dvrp.nodes[r[0]]) >= dvrp.maxdist:
            r.insert(i, 0)
            d = 0
        elif d + distance(dvrp.nodes[r[i]], dvrp.nodes[r[i + 1]]) > dvrp.maxdist:
            r.insert(i, 0)
            d = 0
        else:
            d += distance(dvrp.nodes[r[i]], dvrp.nodes[r[i + 1]])
            i += 1
        #print(f"{d=}, post{r=}")
        #if i > 10:
        #    break
    r = remove_zeros(r)
    return r


def genetic_algorithm(dvrp: GraphDVRP, iterations, popsize):
    FILE = open("data.txt", "w")
    best_route = None
    best_route_value = sys.float_info.max
    # Generate a random initial population
    population = []
    for i in range(popsize):
        p = [i for i in range(1, len(dvrp.nodes))]
        shuffle(p)
        population.append(p)
    # Iterations
    for i in range(iterations):
        population.sort(reverse=False, key=lambda item: fit(dvrp, add_zeros(dvrp, item.copy())))
        next_population = []
        population_len = len(population)
        for j in range(int(population_len / 2)):
            if j % 2 == 0:
                parent1 = population[j]
                parent2 = population[j+1]
            else:
                parent1 = population[j]
                parent2 = population[j + randint(1, int(population_len / 2))]

            # Crossover
            min_len = min(len(parent1), len(parent2))
            cut_point1 = randint(0, min_len-1)
            cut_point2 = cut_point1 + randint(1, min_len-cut_point1)
            cut1 = parent1[cut_point1:cut_point2]
            cut2 = parent2[cut_point1:cut_point2]
            par_list1 = [ele for ele in parent1 if not ele in cut2]
            par_list2 = [ele for ele in parent2 if not ele in cut1]
            child1 = par_list1[:cut_point1] + cut2 + par_list1[cut_point1:]
            child2 = par_list2[:cut_point1] + cut1 + par_list2[cut_point1:]
            next_population += [child1, child2]

        # Mutation - 10% of population will mutate
        mutation_count = int(popsize * 0.1) if popsize > 10 else 1
        for j in range(mutation_count):
            mutated_population = next_population[randint(0, len(next_population) - 1)]
            i1 = randint(0, len(mutated_population) - 1)
            i2 = randint(0, len(mutated_population) - 1)
            mutated_population[i1], mutated_population[i2] = mutated_population[i2], mutated_population[i1]

        mutation_count = int(popsize * 0.1) if popsize > 10 else 1
        for j in range(mutation_count):
            mutated_population = next_population[randint(0, len(next_population) - 1)]
            i1 = randint(0, len(mutated_population) - 1)
            mutated_population.insert(i1, 0)

        for p in population:
            remove_zeros(p)

        population = next_population
        for r in population:
            route = add_zeros(dvrp, r)
            f = fit(dvrp, route)
            if f < best_route_value:
                best_route_value = f
                best_route = remove_zeros(route)
        FILE.write(str(best_route_value) + "\n")
    FILE.close()
    return (best_route, best_route_value)


def draw(dvrp: GraphDVRP, route):
    w, h = 2000, 2000
    img = Image.new("RGB", (w, h), "white")
    draw_img = ImageDraw.Draw(img)
    for i, point in enumerate(dvrp.nodes):
        x, y = point.x * 100 + w / 2, point.y * 100 + h / 2
        color = "blue" if i > 0 else "red"
        draw_img.ellipse((x - 10, y - 10, x + 10, y + 10), fill=color, outline=(0, 0, 0))
        draw_img.text((x, y), str(point.label), fill="black")
    for i in range(len(route) - 1):
        begin = dvrp.nodes[route[i]]
        end = dvrp.nodes[route[i + 1]]
        draw_img.line([(begin.x * 100 + w / 2, begin.y * 100 + h / 2), (end.x * 100 + w / 2, end.y * 100 + h / 2)],
                      fill="black")
    img.show()