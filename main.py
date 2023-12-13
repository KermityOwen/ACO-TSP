from Ants import Ants
from ACO_TSP import ACO_TSP
from EAS_TSP import EAS_TSP
from MMAS_TSP import MMAS_TSP
import numpy as np
import random

if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    np.seterr(divide='ignore')
    
    graph_path = "./TSPLIB_XML/" + input("Graph name here (Graphs stored in ./TSPLIB_XML): ")
    model = int(input("Choose ACO: \n(1 - Vanilla ACO), \n(2 - Elitist AS), \n(3 - MMAS). \nChoice: "))
    
    if model == 1:
        ACO = ACO_TSP(graph_path=graph_path)
    elif model == 2:
        ACO = EAS_TSP(graph_path=graph_path)
    elif model == 3:
        ACO = MMAS_TSP(graph_path=graph_path)
    else:
        print("Selection out of bounds. Please select only options 1-3.")
        exit()
    
    ACO.run()