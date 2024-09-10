from abc import ABC, abstractmethod
from openqaoa import QUBO,QAOA


class QAOASolver:
    '''
    TODO
    '''

    def __init__(self):
        pass

    def get_alo_ising(self,alo,qubo_configuration):
        '''
        TODO
        '''
        alo_ising,_,_ = alo.to_qubo_and_ising(**qubo_configuration)
        
        return alo_ising
    
    def create_and_configure_qaoa(self,device,circuit_configuration,backend_configuration,
                                  ising,optimizer_configuration = None):
        '''
        TODO
        '''
        qaoa = QAOA()
        qaoa.set_device(device)
        qaoa.set_circuit_properties(**circuit_configuration)
        qaoa.set_backend_properties(**backend_configuration)
        if optimizer_configuration is not None:
            qaoa.set_classical_optimizer(**optimizer_configuration)
        qaoa.compile(ising)

        return qaoa
    
    def filter_qubo_solutions(self,alo,qubo_solutions):
        '''
        TODO
        '''
        # only takes the feasible solutions
        final_qubo_solutions = []
        for solution in qubo_solutions.keys():
            print(solution)
            if alo.is_qubo_feasible(solution):
                final_qubo_solutions.append(solution)
        #print(final_qubo_solutions)

        # translate the solutions from QUBO format to standard format and calculate its gain.
        # store the pair (standardSolution,gain) with the maximal gain
        final_standard_gain = float('-inf')
        final_standard_solution = None
        for qubo_solution in final_qubo_solutions:
            standard_solution = alo.quboSolution_to_standardSolution(qubo_solution)
            standard_gain = alo.calculate_standard_gain(standard_solution)
            
            if final_standard_gain < standard_gain:
                final_standard_solution = standard_solution
                final_standard_gain = standard_gain

        return final_standard_solution,final_standard_gain