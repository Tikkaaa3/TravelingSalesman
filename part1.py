#pandas for storing data from tsp
#numpy for storing route

import numpy as np
import pandas as pd


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


file_name = "cities/berlin11_modified.tsp"  # Path to the city file
cities_df = parser(file_name)  # Call the parser function

# Print the resulting DataFrame
print(cities_df)
