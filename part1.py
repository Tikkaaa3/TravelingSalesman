import numpy as np

import travelingsalesman as tsp

# Parse cities
file1 = "cities/berlin11_modified.tsp"
cities_df = tsp.parser(file1)

# Display the solution for a greedy approach
tsp.solution_info(tsp.greedy(cities_df, 1), cities_df)

# Generate initial population
population = tsp.initial_population(cities_df)

# Display population and fitness scores which is a df
population_w_fitness, best_score = tsp.population_info(population, cities_df)

# Extract the elite solutions and update the population DataFrame
elite_solutions = tsp.elite(population_w_fitness)

# Display the elite solutions and the updated DataFrame
print("\nElite Solutions:")
print(elite_solutions)

# print("\nUpdated Population (Without Elite Solutions):")
# print(population_w_fitness)

# Extract the tournament solutions and update the population w fitness Dataframe
tournament_solutions = tsp.tournament(population_w_fitness)

# Display the tournament solutions and the updated DataFrame
print("\nTournament Solutions:")
print(tournament_solutions)

# print("\nUpdated Population (Without Tournament Solutions):")
# print(population_w_fitness)

# Merge elite solutions w tournament solutions to get selected parents
# parents = np.append([np.array(tournament_solutions)], [np.array([elite for elite in elite_solutions])])
parents = np.concatenate((elite_solutions, tournament_solutions))

print("\nParents:")
print(parents)

epoch = 0
best_scores = []
while epoch < 500:
    # reset the pop for creating the new generation
    population = []

    for _ in range(50):
        # Cycle crossover gives me kid1 and kid2 with swap mutation option
        kid1, kid2 = tsp.cycle_crossover(parents)
        population.append(kid1)
        population.append(kid2)

    tsp.population_info(population, cities_df)
    population_w_fitness, best_score = tsp.population_info(population, cities_df)
    best_scores.append(best_score)
    elite_solutions = tsp.elite(population_w_fitness)
    tournament_solutions = tsp.tournament(population_w_fitness)
    parents = np.concatenate((elite_solutions, tournament_solutions))
    epoch += 1
best_scores = np.array(best_scores)
best_score = np.min(best_scores)
mean = np.mean(best_scores)
print("\nBest Solutions:")
print(best_scores)
print("\nMean:")
print(mean)
print("\nBest Score:")
print(best_score)
