import numpy as np
import xml.dom.minidom as xml

class Ants:
    def __init__(self, starting_position=0):
        """ Construtor for Ants

        Args:
            starting_position: Ants starting position. Defaults to 0.
        """
        # Start position saved for ant resetting
        self.start_pos = starting_position
        self.current_pos = starting_position
        self.found_path = []


    def find_path(self, pheramones, visibility):
        """ Finds path for the ant based on pheramones and visibility

        Args:
            pheramones: Adjacency matrix representing graph pheramones
            visibility: Adjacency matrix representing graph visibility
        """
        # Dupe visibility graph to manipulate without affecting other ants during the same epoch
        temp_vis = np.array(visibility)
        
        # Loop for number of cities times
        for n in range(0, len(temp_vis)-1):
            # Set current column of visibilities to 0 to eliminate revisiting nodes
            temp_vis[:, self.current_pos] = 0
            
            # All possible path to travel to from current node and their pheramones
            # Duped to avoid deadlocks
            pher_paths = np.array(pheramones[self.current_pos])
            
            # Heuristic to make higher pheramones exponentially more attractive
            pher_paths = np.power(pher_paths, 2)  
            
            # All possible path to travel to from current node and their visibilities 
            vis_paths = np.array(temp_vis[self.current_pos]) 
            
            # Combining both visibilities and phermones by multiplying each pher by
            combined_paths = np.multiply(pher_paths,vis_paths) 
            
            # Finding the cumulative probability of each node
            total = sum(combined_paths)
            prob_paths = combined_paths/total
            prob_paths = np.cumsum(prob_paths)
            
            # RNG for finding the next node to travel to
            rand = np.random.random_sample()
            
            # Getting 0,0 of the non zero list eliminates the chances of choosing past visited nodes
            next_node = np.nonzero(prob_paths>rand)[0][0]
            
            # print(possible_paths)
            # print(prob_paths)
            # print(rand)
            # print(next_node)
            
            # Adding next node to found path
            self.found_path.append(next_node)
            self.current_pos = next_node
            
        # Last node to travel to complete cycle
        self.found_path.append(self.start_pos)
        self.current_pos = self.start_pos
        
        
    def eval_cost(self, distances):
        """ Evaluate total cost of the stored path from ant

        Args:
            distances: Adjacency matrix representing graph distances

        Returns:
            total_distance: Total cost of stored path
        """
        total_distance = 0
        prev_node = self.start_pos
        for n in self.found_path:
            total_distance += distances[prev_node][n]
            prev_node=n
        # print(total_distance)
        return total_distance


    def eval_pher_update(self, Q, pheramones, distances):
        """ Evaluate pheramones for update based on stored path

        Args:
            Q: Rate of pheramone depositing
            pheramones: Adjacency matrix representing graph pheramones
            distances: Adjacency matrix representing graph distances

        Returns:
            temp_pher: Returns updated adjacency matrix of pheramones
        """
        # Dupes pheramones graph to avoid deadlock
        temp_pher = np.array(pheramones)
        
        # Pheramone to drop off along path
        pher_dropoff = Q/self.eval_cost(distances)
        
        # Loop through the found path 
        prev_node = self.start_pos
        for n in self.found_path:
            # Mirrored for data consistency
            temp_pher[prev_node][n] = temp_pher[prev_node][n] + pher_dropoff
            temp_pher[n][prev_node] = temp_pher[n][prev_node] + pher_dropoff

            prev_node = n
        
        return temp_pher
                
                
    def reset_ant(self):
        """ Resets Ant
        """
        self.current_pos = self.start_pos
        self.found_path = []
        
        
    def __str__(self):
        """ To String

        Returns:
            string: Cute ant descriptor
        """
        return "Ant: [-_-], Current Pos: " + str(self.current_pos)


# DEBUG SHIT
# if __name__ == "__main__":
#     xml_data = xml.parse("brazil58.xml")
#     all_vertex = xml_data.getElementsByTagName("vertex")
        
#     # A nVertex x nVertex sized array filled with 0
#     adj_matrix = np.zeros(shape=(len(all_vertex), len(all_vertex)))
        
#     row_index = 0
#     # Nested loop to go through every edge
#     for vertex in all_vertex:
#         tempVertex = vertex.getElementsByTagName("edge")
#         for edge in tempVertex:
#             # Gets the content in edge which happens to be the column index
#             column_index = int(edge.childNodes[0].nodeValue)
#             cost = edge.getAttribute("cost")
#             # Assigns the cost to the adjacency matrix
#             adj_matrix[column_index, row_index] = cost
#         row_index += 1
        
#     ant = Ants(0)
#     print(ant.eval_cost(adj_matrix))