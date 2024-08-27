import random as r
import json
import numpy as np
import copy


class ALORandomGenerator:
    '''  
    This class implements a random generator for AircraftLoadingOptimization (ALO) instances.
    
    Each instance is formed by:      
        - num_containers
        - num_positions
        - containers_type
            an integers list where element at position i shows the type of container i. 
            The available sizes are:
                * 1 - medium size
                * 2 - small size
                * 3 - big size
        - containers_weight
            a floats list where element at position i shows the weight of container i.
        - t_list
            a floats list for all t_i values, associated to each i container.
            The values are:
                * 0.5 if type(container_i) == 3
                * 1 else
        - d_list
            a floats list for all d_i values, associated to each i container.
            The values are:
                * 0.5 if type(container_i) == 2
                * 1 else
        - max_weight
            The maximum weight supported by the plane.
        - allBinaryVariables:
            a list of all the binary variables for a QUBO formulation  
    '''


    def __init__(self,num_containers=None,num_positions=None,max_weight=100):
        '''
          Parameters (described at the first comment of the class):
            
            - num_containers
            - num_positions
            - max_weight 
            - affinityWeightCoeff (optinal)
          '''
          
        self.num_containers = num_containers
        self.num_positions = num_positions
        self.max_weight = max_weight
        
    def generate_random_instance(self):
        '''
        This method generate a random ALO instance making use of the parameters given at __init__

        Return:
            - a json with the instance attributes
        '''

        containers,containers_weight = self.__create_containers_and_weights()

        containers_type,t_list,d_list = self.__asociate_type_to_containers(containers)

        # prepare the data structure and save it
        allBinaryVariables = list(range(self.num_containers * self.num_positions))
        data = {
            "num_containers": self.num_containers,
            "num_positions": self.num_positions,
            "containers_weight": containers_weight,
            "containers_type":containers_type,
            "t_list":t_list,
            "d_list":d_list,
            "max_weight":self.max_weight,
            "allBinaryVariables":allBinaryVariables
        }
        json_data = json.dumps(data, indent=4)
        return json_data
    
    def __create_containers_and_weights(self):
        '''
        '''
        containers = list(range(self.num_containers))
        containers_weight = [r.randint(0,self.max_weight) for i in containers]

        return containers,containers_weight
    
    def __asociate_type_to_containers(self,containers):
        '''
        '''
        containers_type = [r.randint(1,3) for i in containers]
        t_list = [0.5 if type == 3 else 1 for type in containers_type]
        d_list = [0.5 if type == 2 else 1 for type in containers_type]
        return containers_type,t_list,d_list 
