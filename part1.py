import travelingsalesman as tsp

# Parse cities
file1 = "cities/berlin11_modified.tsp"
cities_df = tsp.parser(file1)

# Display the solution for a greedy approach
tsp.solution_info(tsp.greedy(cities_df, 1), cities_df)

# Generate initial population
population = tsp.initial_population(cities_df)

# Display population and fitness scores which is a df
population_w_fitness = tsp.population_info(population, cities_df)

# Extract the elite solutions and update the population DataFrame
elite_solutions = tsp.elite(population_w_fitness)

# Display the elite solutions and the updated DataFrame
print("\nElite Solutions (Top 2):")
print(elite_solutions)

print("\nUpdated Population (Without Elite Solutions):")
print(population_w_fitness)

# Extract the tournament solutions and update the population w fitness Dataframe
tournament_solutions = tsp.tournament(population_w_fitness)

# Display the tournament solutions and the updated DataFrame
print("\nTournament Solutions:")
print(tournament_solutions)

print("\nUpdated Population (Without Tournament Solutions):")
print(population_w_fitness)
