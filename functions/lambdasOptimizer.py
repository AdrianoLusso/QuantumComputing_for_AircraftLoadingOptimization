from scipy.optimize import minimize
import numpy as np

from openqaoa import QAOA, QUBO
from openqaoa.backends import create_device
from openqaoa.utilities import bitstring_energy
from openqaoa.problems.converters import FromDocplex2IsingModel

import json

class LambdasOptimizer():

    def __init__(self,problemClass,randomGenerator,n_instances) -> None:
        '''
        '''
        self.problemClass = problemClass
        self.randomGenerator = randomGenerator
        self.n_instances = n_instances

        self.init_lambdas = None
        self.random_instances = None
        self.random_mdls = None

        self.optimized_lambdas = None
        self.optimization_message = None

    def optimize(self,init_lambdas = [0.9603,0.0371]):
        '''
        '''
        if init_lambdas == 'random':
                init_lambdas = np.random.uniform(low=-1, high=1.0, size=2)
        self.init_lambdas = init_lambdas

        lambdas_cost_function = self.__create_lambdas_cost_function()

        self.optimization_message = minimize(lambdas_cost_function, init_lambdas, method='Nelder-Mead')
        self.optimized_lambdas = self.optimization_message['x']

        return self.optimization_message
    
    def generate_random_instances(self):
        '''
        '''
        gen = self.randomGenerator

        random_instances = []
        random_mdls = []
        for i in range(self.n_instances):
            instance_dict=gen.generate_random_instance()
            instance_dict = json.loads(instance_dict)
            instance = self.problemClass(instance_dict)

            # WARNING: THIS NAME WORKS PARTICULARLY FOR AIRCRAFT LOADING OPTIMIZATION PROBLEM CLASS
            _,_,mdl = instance.to_qubo_and_ising()
           
            random_instances.append(instance)
            random_mdls.append(mdl)
        
        self.random_instances = random_instances
        self.random_mdls = random_mdls

    def evaluate_lambdas_with_new_instance(self,instance,mdl):
        '''
        '''
        converterWrapper = FromDocplex2IsingModel(mdl, unbalanced_const=True,strength_ineq=self.optimized_lambdas,multipliers=100)
                
        # gets the ising and qubo docplex
        ising = converterWrapper.ising_model
            
        # compile, brute force solve and get energy
        qubo_bruteforce_solver = self.__define_qubo_bruteforce_solver()
        qubo_bruteforce_solver.compile(ising)
        qubo_bruteforce_solver.solve_brute_force(verbose=False)
        fake_opt_energy = qubo_bruteforce_solver.brute_force_results['energy']
    
        cost_hamiltonian = qubo_bruteforce_solver.cost_hamil
        optimal_standard_solution,_ = instance.solve_standard_with_bruteforce()
        opt_qubo_solution = instance.standardSolution_to_quboSolution(optimal_standard_solution)
        opt_energy = bitstring_energy(cost_hamiltonian,opt_qubo_solution) 

        return (opt_energy - fake_opt_energy)**2
    
    
    def __create_lambdas_cost_function(self):
        '''
        '''

        optimal_standard_solutions,_ = self.__solve_random_instances(self.random_instances)
        qubo_bruteforce_solver = self.__define_qubo_bruteforce_solver()

        def cost(params):
            multipliers = 100
            fake_opt_energies = []
            opt_energies = []

            for random_instance,mdl,optimal_standard_solution in zip(self.random_instances,self.random_mdls,optimal_standard_solutions):
                converterWrapper = FromDocplex2IsingModel(mdl, unbalanced_const=True,strength_ineq=params,multipliers=multipliers)
                
                # gets the ising and qubo docplex
                ising = converterWrapper.ising_model
            
                # compile, brute force solve and get energy
                qubo_bruteforce_solver.compile(ising)
                qubo_bruteforce_solver.solve_brute_force(verbose=False)
                fake_opt_energies.append(qubo_bruteforce_solver.brute_force_results['energy'])
    
                cost_hamiltonian = qubo_bruteforce_solver.cost_hamil
                opt_qubo_solution = random_instance.standardSolution_to_quboSolution(optimal_standard_solution)
                opt_energies.append( bitstring_energy(cost_hamiltonian,opt_qubo_solution) )
            
            n = len(optimal_standard_solutions)
            opt_energies = np.array(opt_energies)
            fake_opt_energies = np.array(fake_opt_energies)
            #print(opt_energies)
            #print(fake_opt_energies)
            cost = np.sum( (opt_energies - fake_opt_energies)**2 ) / n
            #print(cost)
            return  cost
        return cost

    def __solve_random_instances(self,random_instances):
        '''
        '''
        optimal_standard_solutions = []
        optimal_gains = []
        for inst in random_instances:
            data = inst.solve_standard_with_bruteforce()
            optimal_standard_solutions.append(data[0])
            optimal_gains.append(data[1])
        
        return optimal_standard_solutions,optimal_gains

    def __define_qubo_bruteforce_solver(self):
        '''
        '''
        q = QAOA()
        qiskit_device = create_device(location='local', name='vectorized')
        q.set_device(qiskit_device)
        q.set_circuit_properties(p=1, param_type='standard', init_type='ramp', mixer_hamiltonian='x')
        q.set_backend_properties(prepend_state=None, append_state=None)
        q.set_classical_optimizer(method='powell', maxfev=1000, tol=0.01,
                                optimization_progress=True, cost_progress=True, parameter_log=True)
        
        return q
