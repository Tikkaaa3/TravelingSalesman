import numpy as np
import pandas as pd
import math


def parser(file_name):

    cities = []
    with open(file_name, "r") as f:

        start = False
        for line in f:
            line = line.strip()

            if line == "NODE_COORD_SECTION":
                start = True
                continue

            if line == "EOF":
                break

            if start:
                if len(line.split()) == 3:
                    node, x, y = line.split()
                    cities.append([int(node), float(x), float(y)])

    cities = pd.DataFrame(cities, columns=["Node", "x_location", "y_location"])
    cities.set_index("Node", inplace=True)
    return cities


def distance(city1, city2):
    x1, y1 = city1["x_location"], city1["y_location"]
    x2, y2 = city2["x_location"], city2["y_location"]
    d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return d


def fitness_function(solution, cities):
    res = 0
    for i in range(0, len(solution) - 1):
        res += distance(cities.loc[solution[i]], cities.loc[solution[i+1]])
    res += distance(cities.loc[solution[-1]], cities.loc[solution[0]])
    return res


def info(solution, cities):
    score = fitness_function(solution, cities)
    print(solution)
    print(score)


def greedy(cities, start):
    solution = np.array([start])
    cities_list = [i for i in range(1, len(cities)+1)]
    city = start
    cities_list.remove(city)
    while len(cities) > 0:
        shortest = float('inf')
        for node in cities_list:
            if node not in solution:
                dist = distance(cities.loc[solution[-1]], cities.loc[node])
                if dist < shortest:
                    shortest = dist
                    city = node
        solution = np.append(solution, city)
        cities_list.remove(city)
        if not cities_list:
            solution = np.append(solution, start)
            break
    return solution
