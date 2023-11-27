from Ants import Ants
import numpy as np
import xml.dom.minidom as xml
import random
from typing import List

class ACO_TSP:
    def __init__ (self, graph_path="graph_path", num_ants=58):
        self.distances = self.parse_graph(graph_path)
        self.pheramones = self.init_pheramones2(self.distances)
        self.ants = self.generate_ants(num_ants) # Array of Ants
        
        # Heuristic 
        self.visibilities = 1/self.distances
        self.visibilities[self.visibilities==np.inf] = 0
        
        # print(self.visibility)
        
        self.max_epoch = 100
        self.decay_rate = 0.5
        
        print("distance graph: " + str(self.distances))
        print("")
        print("pheramones graph: " + str(self.pheramones))
        print("")
        for ant in self.ants:
            print(ant)
        

    def parse_graph(self, path):
        """ Turns XML into a graph represented by an adjacency matrix

        Args:
            path: File path for XML file
            
        Returns:
            adj_matrix: NumPy adjacency matrix representing the graph
        """
        xml_data = xml.parse(path)
        all_vertex = xml_data.getElementsByTagName("vertex")
        
        # A nVertex x nVertex sized array filled with 0
        adj_matrix = np.zeros(shape=(len(all_vertex), len(all_vertex)))
        
        row_index = 0
        # Nested loop to go through every edge
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
        """ Initialise pheramones by duping graph matrix and downscaling it 

        Args:
            graph: Adjacency matrix representing graph distance

        Returns:
            pher_matrix: Adjacency matrix representing graph pheramones
        """
        # A nVertex x nVertex sized array filled with 0
        pher_matrix = np.zeros(shape=(len(graph), len(graph)))
        downscale_constant = 0.002
        
        # Nested loop to go through every element on the matrix
        for i in range(0, len(graph)):
            for j in range(0, len(graph)):
                # Duping and downscaling distance graph as pheramone graph
                pher_matrix[i][j] = graph[i][j]*downscale_constant
                
        return pher_matrix
  
  
    def init_pheramones2(self, graph):
        return np.ones(shape=(len(graph), len(graph)))
        
  
    def generate_ants(self, num_ants) -> List[Ants]:
        arr_ant = []
        for n in range(0, num_ants):
            # ant_start_pos = random.randint(0, len(self.distances))
            ant_start_pos = 0
            arr_ant.append(Ants(starting_position=ant_start_pos))
        return arr_ant


    def step_all(self):
        for ant in self.ants:
            ant.find_path(self.pheramones, self.visibilities)


    def update_pheramones(self):
        for ant in self.ants:
            # print(ant.eval_cost(self.distances))
            self.pheramones = ant.eval_pher_update(1, self.pheramones, self.distances)
            ant.reset_ant()


    def decay_pheramones(self):
        self.pheramones = self.decay_rate * self.pheramones
        
        
    def epoch(self):
        self.step_all()
        self.update_pheramones()
        self.decay_pheramones()


    def eval(self):
        self.ants[0].reset_ant()
        self.ants[0].find_path(self.pheramones, self.visibilities)
        print(self.ants[0].found_path)
        print(self.ants[0].eval_cost(self.distances))
        return self.ants[0].found_path
        
    
    def run(self):
        # print("lel")
        for n in range(0, self.max_epoch):
            self.epoch()
        self.eval()
        # print("pheramones graph: " + str(self.pheramones))
        
        
if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    ACO = ACO_TSP(graph_path="./brazil58.xml")
    ACO.run()

    
### PROBLEMS WITH CODE RN:
### PHEROMONES ARE UPDATED AND DECAY TOO FAST