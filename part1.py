# pandas for storing data from tsp
# numpy for storing route

import numpy as np
import pandas as pd
import math
import travelingsalesman as tsp


file1 = "cities/berlin11_modified.tsp"
# file1 = "cities/berlin52.tsp"
cities_df = tsp.parser(file1)
# print(cities_df)

city_nodes = np.arange(1, len(cities_df) + 1)
np.random.shuffle(city_nodes)
solution = city_nodes

result = tsp.fitness_function(solution, cities_df)

# info(solution, cities_df)
# 5474.001848763631

tsp.info(tsp.greedy(cities_df, 1), cities_df)
# 4543.086327880018

# 8182.1915557256725
# berlin52 best w greedy

# for i in cities_df.index:
#     info(greedy(cities_df, i), cities_df)

# implement little bit elite(%5) and much more tournament or roulette(%95) cycle crossover is better to use
