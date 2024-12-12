# pandas for storing data from tsp
# numpy for storing solutions, population

import numpy as np
import pandas as pd
import math
import travelingsalesman as tsp


file1 = "cities/berlin11_modified.tsp"
cities_df = tsp.parser(file1)


tsp.solution_info(tsp.greedy(cities_df, 1), cities_df)


population = tsp.initial_population(cities_df)
tsp.population_info(population, cities_df)
