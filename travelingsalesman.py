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

    for _ in range(395):
        city_nodes = np.arange(1, len(cities) + 1)
        np.random.shuffle(city_nodes)
        city_nodes = np.append(city_nodes, city_nodes[0])
        population.append(city_nodes)

    # for i in range(1, 6):
    #    solution = greedy(cities, i)
    #    population.append(solution)

    # Generate 5 unique random values from 1 to len(cities)
    random_indices = np.random.choice(range(1, len(cities) + 1), size=5, replace=False)

    # Generate solutions
    for i in random_indices:
        solution = greedy(cities, i)
        population.append(solution)

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

    return population_w_fitness, best_score


def tournament(population_w_fitness):

    tournament_solutions = []
    for _ in range(0, 195):

        # drop function is necessary because I want to select parents once
        # (for example 100 init pop to 50 unique parents)
        tournament_rows = population_w_fitness.sample(n=5)
        winner = tournament_rows.sort_values(by='Fitness').head(1)
        winner_solution = winner['Solution'].iloc[0]
        tournament_solutions.append(winner_solution)
        population_w_fitness.drop(winner.index, inplace=True)
    return tournament_solutions


def elite(population_w_fitness):

    # Sort the DataFrame by fitness in ascending order and select the top two
    elites = population_w_fitness.sort_values(by='Fitness').head(5)
    elite_solutions = [elite for elite in elites['Solution'].values]

    # Remove the elite rows from the original DataFrame
    population_w_fitness.drop(elites.index, inplace=True)

    return elite_solutions


def swap(parent):
    # Randomly select two distinct indices and do not select first or last one
    index_1, index_2 = np.random.choice(np.arange(0, len(parent) - 1), size=2, replace=False)
    if index_1 == 0:
        parent[-1] = parent[index_2]
    elif index_2 == 0:
        parent[-1] = parent[index_1]
    parent[index_1], parent[index_2] = parent[index_2], parent[index_1]
    return parent


def cycle_crossover(parents, mutation_chance=0.02):
    # randomly select 2 parents but not the same(replace=False)
    selected_indices = np.random.choice(len(parents), size=2, replace=False)
    parent_x, parent_y = parents[selected_indices]
    parent_x = parent_x[:-1]
    parent_y = parent_y[:-1]
    random_index = np.random.randint(0, len(parent_x)-1)
    order = []
    index_x = random_index
    val_y = parent_y[index_x]

    while True:
        order.append(index_x)
        for i, val in enumerate(parent_x):
            if val == val_y:
                index_x = i
                val_y = parent_y[index_x]
                break

        if index_x == random_index:
            break

    # NumPy evaluates the assignment from right to left.
    # Without .copy(), the values of parent_x[order]
    # might get modified in-place as soon as parent_y[order] is assigned to it.
    # This can corrupt the values of parent_x[order] before they are assigned to parent_y[order].
    parent_x[order], parent_y[order] = parent_y[order], parent_x[order].copy()
    kid1 = np.append(parent_x, parent_x[0])
    kid2 = np.append(parent_y, parent_y[0])

    # np random random creates a float value between 0 and 1
    if np.random.random() < mutation_chance:
        kid1 = swap(kid1)
        print("Mutation applied to Parent X")
    if np.random.random() < mutation_chance:
        kid2 = swap(kid2)
        print("Mutation applied to Parent Y")

    # we actually created kid1 and kid2
    return kid1, kid2
