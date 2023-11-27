import numpy as np
import xml.dom.minidom as xml

class Ants:
    def __init__(self, starting_position=0):
        self.start_pos = starting_position
        self.current_pos = starting_position
        self.found_path = []

    def find_path(self, pheramones, visibility):
        # Dupe visibility graph to manipulate without affecting other ants during the same epoch
        temp_vis = np.array(visibility)
        
        # Loop for number of nodes times
        for n in range(0, len(temp_vis)-1):
            # Set current pheramone to 0 for no chance of revisiting nodes
            temp_vis[:, self.current_pos] = 0
            
            # All possible path and their pheramones and visibility to travel to from current node
            # Duped to avoid deadlocks
            pher_paths = np.array(pheramones[self.current_pos])
            pher_paths = np.power(pher_paths, 2)  
            
            vis_paths = np.array(temp_vis[self.current_pos])
            vis_paths = np.power(vis_paths, 1)  
            
            combined_paths = np.multiply(pher_paths,vis_paths) 
            
            # Finding the cumulative probability of each node
            total = sum(combined_paths)
            prob_paths = combined_paths/total
            prob_paths = np.cumsum(prob_paths)
            
            # Finding the next node to travel to
            rand = np.random.random_sample()
            
            # Getting 0,0 of the non zero list eliminates the chances of choosing past nodes
            next_node = np.nonzero(prob_paths>rand)[0][0]
            
            # print(possible_paths)
            # print(prob_paths)
            # print(rand)
            # print(next_node)
            
            # Adding next node to found path
            self.found_path.append(next_node)
            self.current_pos = next_node
            
        self.found_path.append(self.start_pos)
        self.current_pos = self.start_pos
        
        # print(self.found_path)
        
    
    def eval_cost(self, distances):
        total_distance = 0
        prev_node = self.start_pos
        for n in self.found_path:
            total_distance += distances[prev_node][n]
            prev_node=n
        return total_distance


    def eval_pher_update(self, Q, pheramones, distances):
        temp_pher = np.array(pheramones)
        delta_distance = Q/self.eval_cost(distances)
        prev_node = self.start_pos
        for n in self.found_path:
            # Mirrored for data consistency
            temp_pher[prev_node][n] = temp_pher[prev_node][n] + delta_distance
            temp_pher[n][prev_node] = temp_pher[n][prev_node] + delta_distance
            # print(temp_pher[prev_node,n])
            
            prev_node = n
        
        return temp_pher
                
                
    def reset_ant(self):
        self.current_pos = self.start_pos
        self.found_path = []
        
        
    def __str__(self):
        return "ant: [-_-], current pos: " + str(self.current_pos)


if __name__ == "__main__":
    xml_data = xml.parse("brazil58.xml")
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
        
    ant = Ants(0)
    print(ant.eval_cost(adj_matrix))