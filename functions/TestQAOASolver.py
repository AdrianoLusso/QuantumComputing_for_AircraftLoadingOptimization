from openqaoa import QAOA
from openqaoa import QUBO
from openqaoa.algorithms import QAOAResult
from openqaoa.backends import create_device
from qiskit_aer import AerSimulator
#import gzip
import shutil
from QAOASolver import QAOASolver

#import sys
#import os
#import asyncio
import json

from ALOClassic import ALOClassic
from ALORandomGenerator import ALORandomGenerator


class TestQAOASolver(QAOASolver):
    '''
    TODO
    '''

    def __init__(self):
        '''
        TODO
        '''
        self.fixedInstances = None
        self.alo_init_configuration = None

    # WORKFLOW
    def sample_workflows_with_arbitraryInstances(self,configuration_name,n_samples,alo_init_configuration,circuit_configuration,
                         optimizer_configuration,optimization_backend_configuration,
                         evaluation_backend_configuration,qubo_configuration,device=None):
        '''
        this method do a sample of 'testQAOAsolver' workflows and save them in json files. The ALO instances are arbitrary, which
        means that they are created randomly during the sample.

        Parameters:
            - configuration_name
                a string with the name of the configuration, so as to identify the json file
            - n_samples            
            - alo_init_configuration
                a dictionary with the parameters for the ALORandomGenerator. This should specify:
                    ^ num_containers: integer
                    ^ num_positions: integer
                    ^ max_weight: boolean
            - circuit_configuration
                a dictionary with the parameters for the set_circuit_properties() method of a QAOA object. This should specify:
                    ^ p
                    ^ param_type
                    ^ init_type
                    ^ mixer_hamiltonian
                more info about possible parameters values in OpenQAOA docs.
            - optimizer_configuration
                a dictionary with the parameters for the set_classical_optimizer() method of a QAOA object. This should specify:
                    ^ method
                    ^ maxfev
                    ^ tol
                    ^ optimization_progress: boolean
                    ^ cost_progress: boolean
                    ^ parameter_log: boolean
                more info about possible parameters values in OpenQAOA docs.
            - optimization_backend_configuration
                a dictionary with the backend optimization to be used during the QAOA optimization
            - evaluation_backend_configuration
                a dictionary with the backend optimization to be used during the QAOA evaluation
            - qubo_configuration
                a dictionary with the parameters for creating the QUBO and Ising through FromDocplex2IsingModel() method.
                    ^ unbalanced: if True, unbalanced approach will be used. Otherway, the typical slack variables approach.
                    ^ lambdas: list of the two lambdas multipliers for the unbalanced approach
                    ^ multipliers:a number, or list of numbers, with the lagrange multipliers for the penalties that are not
                      from the unbalanced approach
            - device
                the device where QAOA will be run.
        '''
        
        # if necessary, create the device
        if device is None:
            device = create_device(location='local', name='qiskit.shot_simulator')

        # create the random ALO instances generator
        alo_gen = ALORandomGenerator(**alo_init_configuration)
        
        # starts the sampling
        samples = {
            'alo_configuration':alo_init_configuration,
            'circuit_configuration':circuit_configuration,
            'optimizer_configuration': optimizer_configuration,
            'optimization_backend_configuration':optimization_backend_configuration,
            'evaluation_backend_configuration':evaluation_backend_configuration
            }
        for sample_index in range(n_samples):
            #create a random instance
            alo_instance_dict = alo_gen.generate_random_instance()
            alo_instance_dict = json.loads(alo_instance_dict)
            alo = ALOClassic(alo_instance_dict)
            
            # get the important results
            print('running sample ',sample_index)
            approximation_ratio,standard_gain_difference, ising_cost_difference,opt_standard_solution, final_standard_solution,qaoa_result = self.__run_workflow(
                alo,circuit_configuration,
                optimizer_configuration,
                optimization_backend_configuration,
                evaluation_backend_configuration,
                qubo_configuration,
                device
            )

            # creates the sample data structure and saves it
            postprocessed_qaoa_result = self.postprocess_qaoaresult(qaoa_result.asdict())
            sample = {
                'instance': alo_instance_dict,
                'approximation_ratio':approximation_ratio,
                'standard_gain_difference':standard_gain_difference,
                'ising_cost_difference':ising_cost_difference,
                'opt_standard_solution':opt_standard_solution,
                'final_standard_solution':final_standard_solution,
                'result':postprocessed_qaoa_result
            }
            samples[sample_index] = sample
            with open('./conf%s.json'%(str(configuration_name)), 'w', encoding='utf-8') as file:
                json.dump(samples, file, ensure_ascii=False, indent=4)
            
            # THIS WILL CREATE A COMPRESED FILE GZ
            #with open('./conf%s.json'%(str(configuration_name)), 'rb') as f_in:
            #    with gzip.open('./conf%s.gz'%(str(configuration_name)), 'wb') as f_out:
            #        shutil.copyfileobj(f_in, f_out)

    def sample_workflows_with_fixedInstances(self,configuration_name,circuit_configuration,
                         optimizer_configuration,optimization_backend_configuration,
                         evaluation_backend_configuration,qubo_configuration,device=None):
        '''
        this method do a sample of 'testQAOAsolver' workflows and save them in json files. The ALO instances are fixed, which
        means that they have to be created before, using set_fixedInstances() method. If these are not previously created, 
        this method won't work.
        Parameters:
            - configuration_name
                a string with the name of the configuration, so as to identify the json file
            - circuit_configuration
                a dictionary with the parameters for the set_circuit_properties() method of a QAOA object. This should specify:
                    ^ p
                    ^ param_type
                    ^ init_type
                    ^ mixer_hamiltonian
                more info about possible parameters values in OpenQAOA docs.
            - optimizer_configuration
                a dictionary with the parameters for the set_classical_optimizer() method of a QAOA object. This should specify:
                    ^ method
                    ^ maxfev
                    ^ tol
                    ^ optimization_progress: boolean
                    ^ cost_progress: boolean
                    ^ parameter_log: boolean
                more info about possible parameters values in OpenQAOA docs.
            - optimization_backend_configuration
                a dictionary with the backend optimization to be used during the QAOA optimization
            - evaluation_backend_configuration
                a dictionary with the backend optimization to be used during the QAOA evaluation
            - device
                the device where QAOA will be run.
        '''
        
        # evaluate if the fixed instances has been previously created
        if self.fixedInstances is None:
            raise ValueError('There fixed instances were not created. Create them using set_fixedInstances() method.')
        
        # if necessary, create the device
        if device is None:
            device = create_device(location='local', name='qiskit.shot_simulator')

        # starts the sampling
        samples = {
            'alo_configuration':self.alo_init_configuration,
            'circuit_configuration':circuit_configuration,
            'optimizer_configuration': optimizer_configuration,
            'optimization_backend_configuration':optimization_backend_configuration,
            'evaluation_backend_configuration':evaluation_backend_configuration
            }        
        for sample_index,alo_instance_dict in enumerate(self.fixedInstances):
            #create the object for the instance dictionary
            alo = ALOClassic(alo_instance_dict)
            
            # get the important results
            print('running sample ', sample_index)
            approximation_ratio,standard_gain_difference, ising_cost_difference,opt_standard_solution, final_standard_solution,qaoa_result = self.__run_workflow(
                alo,circuit_configuration,
                optimizer_configuration,
                optimization_backend_configuration,
                evaluation_backend_configuration,
                qubo_configuration,
                device
            )

            # creates the sample dats structure and saves it
            postprocessed_qaoa_result = self.postprocess_qaoaresult(qaoa_result.asdict())
            sample = {
                'instance': alo_instance_dict,
                'approximation_ratio':approximation_ratio,
                'standard_gain_difference':standard_gain_difference,
                'ising_cost_difference':ising_cost_difference,
                'opt_standard_solution':opt_standard_solution,
                'final_standard_solution':final_standard_solution,
                'result':postprocessed_qaoa_result
            }
            samples[sample_index] = sample
            
            # save the json and compress it
            with open('./conf%s.json'%(str(configuration_name)), 'w', encoding='utf-8') as file:
                json.dump(samples, file, ensure_ascii=False, indent=4)
            
            # THIS WILL CREATE A COMPRESED FILE GZ
            #with open('./conf%s.json'%(str(configuration_name)), 'rb') as f_in:
            #    with gzip.open('./conf%s.gz'%(str(configuration_name)), 'wb') as f_out:
            #        shutil.copyfileobj(f_in, f_out)

    def __run_workflow(self,alo,circuit_configuration, optimizer_configuration,
                       optimization_backend_configuration,
                       evaluation_backend_configuration,qubo_configuration,device):
        '''
        This method runs A 'testQAOAsolver' workflow for a particular ALO instance.

        Parameters:
            - alo
                an ALO object
            - circuit_configuration
                a dictionary with the parameters for the set_circuit_properties() method of a QAOA object. This should specify:
                    ^ p
                    ^ param_type
                    ^ init_type
                    ^ mixer_hamiltonian
                more info about possible parameters values in OpenQAOA docs.
            - optimizer_configuration
                a dictionary with the parameters for the set_classical_optimizer() method of a QAOA object. This should specify:
                    ^ method
                    ^ maxfev
                    ^ tol
                    ^ optimization_progress: boolean
                    ^ cost_progress: boolean
                    ^ parameter_log: boolean
                more info about possible parameters values in OpenQAOA docs.
            - n_shots_for_optimization
                the integer number of shots for the quantum ansatz during the quantum-classical optimization loop
            - n_shots_for_validation
                the integer number of shots for the quantum ansatz during the validation of the optimized variational parameters.
                The number of this parameters will be the number of solutions that will be evaluated as possible solutions.
            - device
                the device where QAOA will be run.

        Return:
            approximation_ratio
                a 0 to 1 ratio showing how approximate to the optimal solution was the solution getted from the QAOA.
            standard_gain_difference
                absolute difference between the gain of the initial standard solution (an empty aircraft) and the final standard
                solution (the containers positioned in the aircraft after QAOA optimization)
            ising_cost_difference
                the absolute difference between the initial ising expectation value and the final ising expectation value
            opt_standard_solution
                optimal solution, in standard formulation, found by brute force
            final_standard_solution
                final solution, in standard formulation, found by QAOA optimization
            qaoa_result
                a QAOA_Result object from OpenQAOA
            

        '''
        # initial configuration
        initial_standard_solution = [[] for i in range(alo.instance_dict['num_positions'])]
        initial_standard_gain = alo.calculate_standard_gain(initial_standard_solution)

        #optimal configuration
        opt_standard_solution,opt_standard_gain = alo.solve_standard_with_bruteforce(debug_every=0)
        
        # final configuration - after QAOA optimization
        final_standard_solution,final_standard_gain,initial_ising_cost,final_ising_cost,qaoa_result = self.__run_qaoa_workflow(
            alo,
            circuit_configuration,
            optimizer_configuration, 
            optimization_backend_configuration,
            evaluation_backend_configuration,
            qubo_configuration,device
        )

        if opt_standard_gain == 0:
            approximation_ratio = None
        else:
            approximation_ratio = final_standard_gain / opt_standard_gain
        standard_gain_difference = final_standard_gain - initial_standard_gain
        ising_cost_difference = final_ising_cost - initial_ising_cost
        
        # TODO convegence curve 

        return approximation_ratio,standard_gain_difference,ising_cost_difference,opt_standard_solution,final_standard_solution,qaoa_result

    def __run_qaoa_workflow(self,alo,circuit_configuration, optimizer_configuration, optimization_backend_configuration,
                       evaluation_backend_configuration,qubo_configuration,device):
        '''
        this  methods runs the QAOA workflow (sub-workflow of the complete 'testQAOASolver') for a particular ALO instance

        Parameters:
            -alo
                a ALO object
            - circuit_configuration
                a dictionary with the parameters for the set_circuit_properties() method of a QAOA object. This should specify:
                    ^ p
                    ^ param_type
                    ^ init_type
                    ^ mixer_hamiltonian
                more info about possible parameters values in OpenQAOA docs.
            - optimizer_configuration
                a dictionary with the parameters for the set_classical_optimizer() method of a QAOA object. This should specify:
                    ^ method
                    ^ maxfev
                    ^ tol
                    ^ optimization_progress: boolean
                    ^ cost_progress: boolean
                    ^ parameter_log: boolean
                more info about possible parameters values in OpenQAOA docs.
            - n_shots_for_optimization
                the integer number of shots for the quantum ansatz during the quantum-classical optimization loop
            - n_shots_for_validation
                the integer number of shots for the quantum ansatz during the validation of the optimized variational parameters.
                The number of this parameters will be the number of solutions that will be evaluated as possible solutions.
            - device
                the device where QAOA will be run.

        Return:
            - final_standard_solution
                final solution, in standard formulation, found by QAOA optimization
            - final_standard_gain
                the gain asociated to the final standard solution
            - initial_ising_cost
            - final_ising_cost
            - qaoa_result
                a QAOA_Result object from OpenQAOA
        '''
        # gets the ising of the alo
        alo_ising = super().get_alo_ising(alo,qubo_configuration)

        # creates and configure the qaoa for optimization
        qaoa = super().create_and_configure_qaoa(device,circuit_configuration,
                                          optimization_backend_configuration,
                                          alo_ising,
                                          optimizer_configuration)
        
        #compile
        qaoa.compile(alo_ising)
        # if the size of the ALO problem is of 30 or more qubits, change the precision to single
        #if len(alo.instance_dict['allBinaryVariables']) >29:
        #    qaoa.backend.backend_simulator = AerSimulator(precision='single')

        # get the initial ising cost, using the initial variational parameters
        initial_ising_cost = qaoa.evaluate_circuit(qaoa.variate_params.asdict())['cost']

        # do the QAOA optimization
        qaoa.optimize()
        qaoa_result = qaoa.result
        
        # get the final ising cost, using the optimized variational parameters
        final_ising_cost = qaoa_result.optimized['cost']

        # creates and configure the qaoa for evaluation
        qaoa = super().create_and_configure_qaoa(device,circuit_configuration,
                                          evaluation_backend_configuration,
                                          alo_ising)
        preliminary_qubo_solutions = qaoa.evaluate_circuit(qaoa_result.optimized['angles'])['measurement_results']

        # filter the qubo solutions, getting the final standard solution and gain
        final_standard_solution,final_standard_gain = super().filter_qubo_solutions(
            alo,
            preliminary_qubo_solutions)
        
        return final_standard_solution,final_standard_gain,initial_ising_cost,final_ising_cost,qaoa_result    

    def set_fixedInstances(self,alo_init_configuration,n_instances):
        '''
        TODO
        '''
        # save the ALO init configuration
        self.alo_init_configuration = alo_init_configuration
        
        # create the random ALO instances generator
        alo_gen = ALORandomGenerator(**alo_init_configuration)

        # create and save the dictionaries of the fixed instances
        self.fixedInstances = []
        for _ in range(n_instances):
            alo_instance_dict = alo_gen.generate_random_instance()
            alo_instance_dict = json.loads(alo_instance_dict)
            self.fixedInstances.append(alo_instance_dict)

    def delete_fixedInstances(self):
        '''
        TODO
        '''
        self.fixedInstances = None
        self.alo_init_configuration = None

    # AUXILIARY
    def postprocess_qaoaresult(self,result_dict):
        '''
        TODO
        '''
        cost_history = result_dict['intermediate']['cost']
        result_dict.pop('intermediate')
        result_dict['cost_history'] = cost_history

        return result_dict
