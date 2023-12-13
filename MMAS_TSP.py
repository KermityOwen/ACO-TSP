from Ants import Ants
import numpy as np
import xml.dom.minidom as xml
import random
import matplotlib.pyplot as plot
from typing import List
from os.path import isfile

class MMAS_TSP:
    def __init__ (self, graph_path="./TSPLIB_XML/brazil58.xml"):
        """Constructor for MMAS_TSP class

        Args:
            graph_path: path to TSPLIB in XML file. Defaults to "graph_path".
        """
        # Parameters
        self.max_epoch = 300
        self.decay_rate = 0.9
        self.num_ants = 100
        self.dropoff_rate = 1
        
        self.max = 100000000
        self.min = 0.00000001
        
        # Ensure file path is correct
        if not isfile(graph_path):
            print("File does not exist in path entered.")
            print("Please ensure the file name was entered correctly. \nProgram exited.")
            exit()
        
        # Adjacency matrices for distance, pheramones, and our heuristic (visibility)
        self.distances = self.parse_graph(graph_path)
        self.pheramones = self.init_pheramones(self.distances)
        self.visibilities = self.init_visibilities(self.distances) # Our heuristic 
        
        # Generates an array of ants
        # Scatter boolean determines if the ants should all start at 0, or at randomised cities
        self.ants = self.generate_ants(self.num_ants, scatter=True)
        
        # Initialising current best path
        self.current_best_path = []
        self.current_best_dis = 999999999
        
        self.converged_at = 0
        
        
    def print_self(self):
        """ Overrides to_string function
        """
        print("Distances graph: " + str(self.distances))
        print("")
        print("Pheramones graph: " + str(self.pheramones))
        print("")
        print("Heuristics graph: " + str(self.visibilities))
        print("")
        
        # print(str(self.ants[0]) + " x", str(self.num_ants))
        for ant in self.ants:
            print(ant)


    def parse_graph(self, path):
        """ Turns XML into a graph represented by an adjacency matrix

        Args:
            path: File path for XML file
            
        Returns:
            adj_matrix: NumPy adjacency matrix representing the graph
        """
        # Parse xml into readable data
        xml_data = xml.parse(path)
        all_vertex = xml_data.getElementsByTagName("vertex")
        
        # A nVertex x nVertex sized array filled with 0
        adj_matrix = np.zeros(shape=(len(all_vertex), len(all_vertex)))
        
        # Nested loop to go through every edge
        row_index = 0
        for vertex in all_vertex:
            tempVertex = vertex.getElementsByTagName("edge")
            for edge in tempVertex:
                # Gets the content in edge which happens to be the column index
                column_index = int(edge.childNodes[0].nodeValue)
                cost = edge.getAttribute("cost")
                # Assigns the cost to the adjacency matrix
                adj_matrix[column_index, row_index] = cost
            row_index += 1
        
        return adj_matrix
        
  
    def init_pheramones(self, graph):
        """ Initialise pheramones by generating and filling a nVertex x nVertex sized matrix by 1s

        Args:
            graph: Adjacency matrix representing graph distance

        Returns:
            pher_matrix: Adjacency matrix representing graph pheramones
        """
        return np.ones(shape=(len(graph), len(graph)))
    
    
    def init_visibilities(self, graph):
        """ Initialise visibility by generating a scaled down distance graph by an exponent of -1

        Args:
            graph: Adjacency matrix representing graph distance

        Returns:
            visibilities: Adjacency matrix representing graph visibility
        """
        # Visibility is distance scaled down exponentially (pow of -1)
        visibilities = 1/graph
        # Ensures if distance is 0, the visibility is also 0 and not 1/0 which is approxed to infinity
        visibilities[visibilities==np.inf] = 0
        return visibilities
        
  
    def generate_ants(self, num_ants, scatter=False) -> List[Ants]:
        """ Generate an array of Ants

        Args:
            num_ants: Number of ants
            scatter: Whether or not to scatter the ants position. Defaults to False.

        Returns:
            arr_ant: Array of Ants
        """
        arr_ant = []
        for n in range(0, num_ants):
            if(scatter):
                # If scatter is True, randomise the starting position for the ants
                ant_start_pos = random.randint(0, len(self.distances)-1) 
            else:
                # If not all the starting position for all the ants are 0
                ant_start_pos = 0
            arr_ant.append(Ants(starting_position=ant_start_pos))
        return arr_ant


    def step_all(self):
        """ Step all ants forward and calculates their found paths
        """
        for ant in self.ants:
            ant.find_path(self.pheramones, self.visibilities)


    def update_best_pher(self):
        """ Update current best path based on all ants traversed path 
            and updates pheramones as well due to this being Elitist
        """
        # Sets initial cost as first ant
        initial_cost = self.ants[0].eval_cost(self.distances)
        
        # Sets local best distance
        local_best = initial_cost
        
        # Initialises best_ant as the first ant of the list
        best_ant = self.ants[0]
        
        # How many ants have converged
        counter = 0
        
        # Loops through every ant in the colony
        for ant in self.ants:
            # Cost has to be used and compared multiple times later
            cost = ant.eval_cost(self.distances)
            
            # If cost is greater than GLOBAL best path replace the current best path and distance
            if cost < self.current_best_dis:
                self.current_best_path = ant.found_path
                self.current_best_dis = cost
            
            # if the current cost beats the local best then sets the local best as current cost
            # and the best ant as current ant
            if cost <= local_best:
                local_best = cost
                best_ant = ant    
            
            # If the ants find the same path increment counter
            if initial_cost == cost:
                counter += 1
        
        # Updates pheramones based on best_ants path and distance
        self.pheramones = best_ant.eval_pher_update(self.dropoff_rate, self.pheramones, self.distances)
        
        # If more than 90% of ants are on the same path, it is considered converged
        if counter > self.num_ants * 0.9:
            return True
        return False


    def reset_ants(self):
        """ Resets all ants
        """
        for ant in self.ants:
            ant.reset_ant()
            
    
    def min_max_pheramones(self):
        """ Rebounds pheramones back to maximum and minimum
        """
        self.pheramones = np.clip(self.pheramones, self.min, self.max)


    def decay_pheramones(self):
        """ Decays / evaporates pheramones by decay rate
        """
        self.pheramones = self.decay_rate * self.pheramones
                
                
    def epoch(self):
        """ Run a single iteration / epoch of the ACO for TSP
        """
        self.step_all()
        is_converged = self.update_best_pher()
        self.reset_ants()
        self.decay_pheramones()
        self.min_max_pheramones()
        return is_converged


    def eval(self):
        """ Generates an ant and let it find a path based on the pheramones and heuristic for evaluation

        Returns:
            found_path: Array representing the found path
            eval_cost: Number representing the total cost (distance) of the found path
        """
        evaluator_ant = Ants(0)
        evaluator_ant.find_path(self.pheramones, self.visibilities)
        return evaluator_ant.found_path, evaluator_ant.eval_cost(self.distances)
        
        
    def eval2(self):
        """ Evaluates based on current best stored path and current best stored distance
            This is NOT The settled local minima but instead a path found previously as the ants traversed

        Returns:
            found_path: Array representing the found path
            eval_cost: Number representing the total cost (distance) of the found path
        """
        return self.current_best_path, self.current_best_dis
    
    
    def run(self):
        """ Run max_epoch number of iterations of the ACO
        """
        # Set counter for when the colony has converged at max_epoch
        converged_n = self.max_epoch
        
        # Running max_epoch iterations of an epoch
        for n in range(0, self.max_epoch):
            is_converged = self.epoch()
            # If it is converged break out to save processing power / time
            if (is_converged):
                converged_n = n
                break
            
        found_path, eval_cost = self.eval2()
        self.converged_at = converged_n
        print("==== MMAS ====")
        print("Final Path: " + str(found_path))
        print("Total Distance: " + str(eval_cost))
        print("Converged at (process killed at): " + str(converged_n))
        
        
if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    np.seterr(divide='ignore')
    # graph_path = "./TSPLIB_XML/burma14.xml"
    # graph_path = "./TSPLIB_XML/brazil58.xml"
    graph_path = "./TSPLIB_XML/" + input("Graph name here (Graphs stored in ./TSPLIB_XML): ")
    print("")
    MMAS = MMAS_TSP(graph_path=graph_path)
    MMAS.run()

