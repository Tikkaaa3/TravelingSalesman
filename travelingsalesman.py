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


def solution_info(solution, cities):
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


def initial_population(cities):
    population = []

    for _ in range(17):
        city_nodes = np.arange(1, len(cities) + 1)
        np.random.shuffle(city_nodes)
        city_nodes = np.append(city_nodes, city_nodes[0])
        population.append(city_nodes)

    for i in range(1, 4):
        solution = greedy(cities, i)
        population.append(solution)

    population = np.array(population)
    return population


def population_info(population, cities):
    fitness_scores = []
    for solution in population:
        fitness_scores.append(fitness_function(solution, cities))

    population_w_fitness = pd.DataFrame({'Solution': [solution for solution in population],
                                         'Fitness': fitness_scores})
    fitness_scores = np.array(fitness_scores)
    best_score = np.min(fitness_scores)
    median_score = np.median(fitness_scores)

    print("Population Information (Solutions with Fitness Scores):")
    print(population_w_fitness)
    print(f"\nBest score: {best_score}")
    print(f"\nMedian score: {median_score}")

    return population_w_fitness


def tournament(population_w_fitness):

    tournament_solutions = []
    for _ in range(0, 8):
        tournament_rows = population_w_fitness.sample(n=4)
        winner = tournament_rows.sort_values(by='Fitness').head(1)
        winner_solution = winner['Solution'].iloc[0]
        tournament_solutions.append(winner_solution)
        population_w_fitness.drop(winner.index, inplace=True)
    return np.array(tournament_solutions)


def elite(population_w_fitness):

    # Sort the DataFrame by fitness in ascending order and select the top two
    elites = population_w_fitness.sort_values(by='Fitness').head(2)
    elite_solutions = elites['Solution'].values

    # Remove the elite rows from the original DataFrame
    population_w_fitness.drop(elites.index, inplace=True)

    return np.array(elite_solutions)


def cycle_crossover(population, cities):
    return
