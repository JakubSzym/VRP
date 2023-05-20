import csv
import random
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


def fit(dvrp: GraphDVRP, route):
    d = max(0, route.count(0) - (dvrp.trucks + 1)) * 999999999
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


def adjust(dvrp: GraphDVRP, route):
    route.insert(0, 0)
    route = remove_zeros(route)
    repeated = True
    while repeated:
        repeated = False
        for i in range(len(route)):
            for j in range(i):
                if route[i] == route[j]:
                    all = True
                    for id in range(len(dvrp.nodes)):
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


def add_zeros(dvrp: GraphDVRP, route):
    i = 0
    d = 0.0
    route_len = len(route)
    cap = dvrp.maxdist
    route.insert(0, 0)
    route.append(0)

    while i < len(route) - 1:
        d += distance(dvrp.nodes[route[i]], dvrp.nodes[route[i + 1]])
        if d > cap:
            route.insert(i, 0)
            d = 0.0
        i += 1
        assert i < route_len * 3, f"Max distance is too small"

    route = remove_zeros(route)
    return route


def genetic_algorithm(dvrp: GraphDVRP, iterations, popsize):
    best_route = None
    best_route_value = sys.float_info.max
    # generage a random initial population
    population = []

    for i in range(popsize):
        p = [i for i in range(1, len(dvrp.nodes))]
        random.shuffle(p)
        population.append(p)

    for i in range(iterations):
        zero_population = []
        for p in population:
            zero_population.append(add_zeros(dvrp, p.copy()))

        fit_population = [fit(dvrp, zero) for zero in zero_population]

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

            cut_point1, cut_point2 = random.randint(1, min(len(parent1), len(parent2)) - 1), random.randint(1, min(len(
                parent1), len(parent2)) - 1)
            cut_point1, cut_point2 = min(cut_point1, cut_point2), max(cut_point1, cut_point2)

            child1 = parent1[:cut_point1] + parent2[cut_point1:cut_point2] + parent1[cut_point2:]
            child2 = parent2[:cut_point1] + parent1[cut_point1:cut_point2] + parent2[cut_point2:]
            next_population += [child1, child2]

            # Mutation

        # 0 - 1% of population will mutate
        mutation_count = random.randint(0, popsize * 0.01)
        for j in range(mutation_count):
            mutated_population = next_population[random.randint(0, len(next_population) - 1)]
            i1 = random.randint(0, len(mutated_population) - 1)
            i2 = random.randint(0, len(mutated_population) - 1)
            mutated_population[i1], mutated_population[i2] = mutated_population[i2], mutated_population[i1]

        for j in range(len(next_population)):
            next_population[j] = adjust(dvrp, next_population[j])
        population = next_population

        for route in population:
            f = fit(dvrp, add_zeros(dvrp, route))
            if f < best_route_value:
                best_route_value = f
                best_route = route
    print(f"{best_route_value=}")
    return best_route


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