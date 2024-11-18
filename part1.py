# pandas for storing data from tsp
# numpy for storing route

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
    return cities


def distance(city1, city2):
    x1 = city1["x_location"]
    x2 = city2["x_location"]
    y1 = city1["y_location"]
    y2 = city2["y_location"]
    d = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return d


file1 = "cities/berlin11_modified.tsp"
cities_df = parser(file1)

print(cities_df)
two_cities_distance = distance(cities_df.loc[0], cities_df.loc[1])
print(two_cities_distance)

cities_df = parser(file_name)  # Call the parser function

# Print the resulting DataFrame
print(cities_df)
