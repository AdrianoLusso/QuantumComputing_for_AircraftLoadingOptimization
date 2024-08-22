
import itertools
import sys
import ipywidgets as widgets

class ALOClassic:    
    '''  
    This class implements the Aircraft Loading Optimization (ALO) and a couple of important methods for its classical
    representantion, calculus and solving.  
    
    FORMALISMS AND FORMATS CONVERSION 

        - standard form: 
            An optimization problem with objetive function and constraints functions. 
            It's expressed as a maximization problem.

        - QUBO form:
            a quadratic unconstrained uptimization problem. As the formalism name says, it only accepts
            binary variables and the constraints are expressed as 'penalties' inside the objective function.
            It is expressed as a minimization problem, which would be later convenient for quantum algorithms.
            Each  binary variable x_{1*i+j} represents if the container 'i' is asigned to the plane position 'j'.

    
        - openqaoa_format:
            a qubo problem, but with a different code implementation. The terms and weights of the function
            are saved in different lists. This format allows using QUBO() class from OpenQAOA.
    '''

    def __init__(self,instance_dict,qubo_terms_weights=None):
          '''
          Parameters:
            instance_dict:
                A dictionary with the instance configuration. 
                This dictionary should be previously created with ALORandomGenerator class,
                or by manually defining its elements.
            qubo_terms_weights (optional):
                In case you alredy has the qubo terms and weights, you could pass them to the ALO object.
                It is important that:
                    1. The qubo_terms_weights represents the same instance that the instance_dict
                    2. The qubo_terms_weights follows a valid structure. As reference, follows the 
                    structure return by to_qubo method
          '''
          self.instance_dict = instance_dict
          self.qubo_terms_weights = qubo_terms_weights

    

    ''' REPRESENTATIONS '''
    def to_qubo(self,debug=False):
        '''
        This method transforms the instance_dict of the ALO object into a qubo_terms_weights dictionary.
        This qubo dictionary forms as follows:
            {
                0: 2.32,
                1: -4.2,
                ...
                (0, 1):4,
                ...
            }
        
        Parameters:
            - debug (optional):
                If True, it shows debug messages about the QUBO construction
        
        Return:
            - The qubo_terms_weights dictionary, which is also saves as an object property

        TODO:
            this method doesn't make use of the differences matrixes for the QUBO construction. Making use
            of them would result in a faster construction.
        '''
       
        # take each (key,value) of the instance dict and form them as variables 'key = value;'
        for key, value in self.instance_dict.items():
            globals()[key] = value

        # the dict that will store the QUBO
        terms_weights = {}

        # calculate the 1 variable terms, related to the original objective function
        # for each variable x_ij, for agent vacancy job i and agent j, its weighted priority and affinity are calculated 
        for i in range(len(vacnJobs)):
            for j in range(len(agents)):        
                finalPriority = priorityWeightCoeff * (vacnJobs_priority[i] - assgJobs_priority[agents_assgJobs[j]])
                finalAffinity = affinityWeightCoeff * (agents_vacnJobsAfinnity[j][i] - agents_assgJobsAfinnity[j][agents_assgJobs[j]])
                
                # the -1 factor is for making a MINIMIZATION problem
                finalWeight = -1*(finalPriority + finalAffinity)

                terms_weights[(allBinaryVariables[len(agents)*i+j])] = finalWeight

        if debug:
            print('----------------------------------------')
            print('After 1 variable terms')
            print('terms:',list(terms_weights.keys()))
            print('weights: ',list(terms_weights.values()))
            print('terms_weights: ',terms_weights)
            print()

        # calculate the first penalty:
        for i in range(len(vacnJobs)):
            for j in range(len(agents)):
                terms_weights[len(agents)*i+j] = terms_weights[len(agents)*i+j] + penalty1*(1 - 2 * 0.5)
                for r in range(j+1,len(agents)):
                    terms_weights[(allBinaryVariables[len(agents)*i+j],allBinaryVariables[len(agents)*i+r])] = 2*penalty1

        if debug:
            print('----------------------------------------')
            print('After first penalty')
            print('terms:',list(terms_weights.keys()))
            print('weights: ',list(terms_weights.values()))
            print('terms_weights: ',terms_weights)
            print()


        for j in range(len(agents)):
            for i in range(len(vacnJobs)):
                terms_weights[len(vacnJobs)*j+i] = terms_weights[len(vacnJobs)*j+i] + penalty2*(1 - 2 * 0.5)
                for r in range(i+1,len(vacnJobs)):
                    key = (allBinaryVariables[len(agents)*i+j],allBinaryVariables[len(agents)*r+j])
                    if key in terms_weights:
                        terms_weights[(allBinaryVariables[len(agents)*i+j],allBinaryVariables[len(agents)*r+j])] += 2*penalty2
                    else:
                        terms_weights[(allBinaryVariables[len(agents)*i+j],allBinaryVariables[len(agents)*r+j])] = 2*penalty2

        if debug:
            print('----------------------------------------')
            print('After second penalty')
            print('terms:',list(terms_weights.keys()))
            print('weights: ',list(terms_weights.values()))
            print('terms_weights: ',terms_weights)
            print()


        # it returns the the qubo terms and weights, but also save it in the instance
        self.qubo_terms_weights = terms_weights
        return terms_weights
    
    def to_openqaoa_format(self,terms_weights=None,debug=False):
        '''
        This method takes a qubo problem a terms_weights dictionary and transform it into openqaoa format,
        which are two different lists for terms and weights.
         
        Parameters:
            - terms_weights (optional)
                the terms_weights list of qubo formulation. If it is not passed, the inner qubo_terms_weights
                list of the JRP object will be used.
            - debug (optional)
                If True, it shows debug messages about the OpenQAOA format adaptation
           
        Return:     
            - The two resulting lists 'terms' and 'weights', which can be used as input for
              creating a QUBO() in OpenQAOA.
        '''
        # it gets the terms_weights for the conversion
        if terms_weights is None:
            terms_weights = self.qubo_terms_weights

        terms = list(terms_weights.keys())
        weights = list(terms_weights.values())

        terms_aux = []
        for term in terms:
            if type(term) == tuple:
                terms_aux.append([term[0],term[1]])
            else:
                terms_aux.append([term])
        terms = terms_aux
        if debug:
            print('final terms: ',terms)
            print('final weights: ',weights)

        return terms,weights

    def quboSolution_to_standardSolution(self,qubo_solution,check_feasibility = False):
        '''
        '''
        # the solution is divide in subsolutions, where each subsolution is a vacant job and its J possible agents.
        if check_feasibility and not self.is_qubo_feasible(qubo_solution):
           raise ValueError('The qubo solution is not feasible')
        
        n = self.instance_dict['num_agents']
        subsolutions = [qubo_solution[i:i+n] for i in range(0, len(qubo_solution), n)]

        standard_solution = [-1 for i in range(self.instance_dict['num_agents'])]
        for vancJob,subsolution in enumerate(subsolutions):
            agent = subsolution.find('1')
            if agent != -1:
                standard_solution[agent] = vancJob
        return standard_solution


    ''' CALCULATIONS '''
    # TODO 
    def calculate_standard_gain(self,solution,minimization=False):
        ''' 
       this method calculate the gain of a solution for the standard (constrained) JRP optimization function

        Parameters:
            - solution
                the solution given should be given in a standard form. That is:
                a list [a_0 , a_1 , a_2 , ... , a_n] where each variable a_i must be a value from [-1,m),
                where m is the num_vacnJobs. If a_i = x | (0 <= x < m), then agent i was reassigned to
                vacant job x. If a_i = -1, then the agent  wasn't reasigned and keep its original job.

            - minimization (optional)
                if True, the standard JRP evaluated will be treated as a minimization problem. By default, is
                False becase the original JRP formulation is as a maximization problem. I case of minimization,
                the 'gain' should be mentioned as 'cost' 

        Return:
            - the gain of the solution given for the standard JRP
        '''        

        #veryfing that len of solution is the same as num_agents
        if(len(solution) != self.instance_dict['num_agents']):
            raise ValueError('The length of the solution must be equal to num_agents = ',self.instance_dict['num_agents'])

    
        weighted_priority_diff_matrix = self.instance_dict['weighted_priority_diff_matrix']
        weighted_affinity_diff_matrix = self.instance_dict['weighted_affinity_diff_matrix']

        # gain calculus
        gain = 0
        alredy_choosen_vacnJobs = []
        for agent,job in enumerate(solution):
            if job != -1:
                if job in alredy_choosen_vacnJobs:
                    raise ValueError('Your solution does not respect the constraint that a vacant job must be assigned to not more than one agent')
                gain += weighted_priority_diff_matrix[job][agent] + weighted_affinity_diff_matrix[job][agent]
                alredy_choosen_vacnJobs.append(job)

        # if its necessary, translate from maximization problem gain to minimization problem cost
        if minimization:
            gain *= -1
        return gain
    
    def is_qubo_feasible(self,solution):
        '''TODO '''
        solution = str(solution)
        if (
            len(solution) == self.instance_dict['num_agents'] * self.instance_dict['num_vacnJobs']
            and set(solution).issubset({'0', '1'})
            and not self.__violate_first_constraint_for_qubo(solution)
            and not self.__violate_second_constraint_for_qubo(solution)            
        ):
            return True
        else:
            return False


    ''' SOLVERS '''
    def solve_standard_with_bruteforce(self,debug_every=0,minimization=False):
        ''' 
        this method solve the standard (constrained) JRP optimization function through brute force

        Parameters:
            - debug_every (optional)
                an integer X that for 'printing a debug message every X combinations analyzed'. If 0,
                no message will be print. If 1, all combinations status will be shown
            minimization (optional)
                if True, the standard JRP evaluated will be treated as a minimization problem. By default, is
                False becase the original JRP formulation is as a maximization problem. I case of minimization,
                the 'gain' should be mentioned as 'cost' 

        Return:
            - the optimal solution and the optimal gain
        '''   
        output_area = None
        if debug_every != 0:
            output_area = widgets.Output()
            display(output_area)
            


        weighted_priority_diff_matrix = self.instance_dict['weighted_priority_diff_matrix']
        weighted_affinity_diff_matrix = self.instance_dict['weighted_affinity_diff_matrix'] 

        # calculate the search space
        combinations = itertools.product(range(-1,self.instance_dict['num_vacnJobs']),
                                         repeat=self.instance_dict['num_agents'])
        total_combinations = (self.instance_dict['num_vacnJobs']+1)**self.instance_dict['num_agents']
        # do the brute force 
        opt_gain = None
        opt_solution = None
        for itr_combination,combination in enumerate(combinations):
            # if the combination doesn't respect the first constraint, it is not evaluated
            if self.__violate_one_agent_per_job(combination):
                self.__short_debug_solve_standard_with_bruteforce(debug_every,itr_combination,
                                                        output_area,total_combinations,
                                                        combination)
                continue
            # the combination gain is calculated
            gain = self.calculate_standard_gain(combination,minimization)
            
            # if the current gain calculated is the optimal one, it is saved, as well for the combination
            if (opt_gain is None 
                or (not minimization and opt_gain < gain)
                or (    minimization and opt_gain>gain)):
                opt_gain = gain
                opt_solution = combination

            # debug prints
            self.__complete_debug_solve_standard_with_bruteforce(debug_every,itr_combination,
                                                        minimization,output_area,total_combinations,
                                                        combination,gain,opt_gain,opt_solution)
        
        if debug_every != 0:
            output_area.close()
        return list(opt_solution),opt_gain
            
    ''' AUXILIAR '''
    def __violate_one_agent_per_job(self,solution):
        '''
        this method return if, given a solution, this violates the first constraint:
            each vacant job can be done by at most one agent

        Parameters:
            - solution
                a solution to evaluate

        Return:
            a boolean indicating if the constraint is beeing violated

        '''
        seen_jobs = set()
        for job in solution:
            if job != -1 and job in seen_jobs:
                return True
            seen_jobs.add(job)
        return False

    def __violate_first_constraint_for_qubo(self,solution):
        ''' 
        this method returns if, given a qubo solution, this violates the first constraint:
            each vacant job can be done by at most one agent

        Parameters:
            - solution
                a qubo solution to evaluate
        
        Return:
            a boolean indicating if the constraint is beeing violated

        '''
        # the solution is divide in subsolutions, where each subsolution is a vacant job and its J possible agents.
        n = self.instance_dict['num_agents']
        subsolutions = [solution[i:i+n] for i in range(0, len(solution), n)]
        
        # for each subsolution, it evaluates if no one has more than one char '1'. If that happens, that means that
        # a vacant job was assigned to more than one agent, violating the constraint
        for subsolution in subsolutions:
            if subsolution.count('1') > 1:
                return True
        
        return False
    def __violate_second_constraint_for_qubo(self,solution):
        ''' 
        this method returns if, given a qubo solution, this violates the second constraint:
            each agent can be reasigned to at most one vacant job

        Parameters:
            - solution
                a qubo solution to evaluate
        
        Return:
            a boolean indicating if the constraint is beeing violated

        '''
        # the solution is divide in subsolutions, where each subsolution is a vacant job and its J possible agents.
        n = self.instance_dict['num_agents']
        subsolutions = [solution[i:i+n] for i in range(0, len(solution), n)]

        # for each subsolution, it evaluates if an agent alredy reasigned is marked with 1, violating the constraint.
        already_reasigned_agents = []
        for subsolution in subsolutions:
            agent = subsolution.find('1')
            if agent != -1 and agent in already_reasigned_agents:
                return True
            else:
                already_reasigned_agents.append(agent)
        
        return False
    
    def __complete_debug_solve_standard_with_bruteforce(self,debug_every,itr_combination,
                                                        minimization,output_area,total_combinations,
                                                        combination,gain,opt_gain,opt_solution):
        if (debug_every > 0 
            and (
                (itr_combination+1) % debug_every == 0 
                or (itr_combination+1) == 1
                or (itr_combination+1) == total_combinations)
            ):
                if minimization:
                    text = 'cost'
                else:
                    text = 'gain'
                #if itr_combination != 0:
                #    self.__clear_previous_lines(8)
                with output_area:
                    output_area.clear_output(wait=True)
                    print('==========================================================')
                    print('ITERATION ',itr_combination+1,' OF ', total_combinations)
                    print('current combination: ',combination)
                    print('current ',text,': ',gain)
                    print()
                    print('current optimal ',text,': ',opt_gain)
                    print('current optimal solution: ',opt_solution)
                    print('==========================================================\n')

    def __short_debug_solve_standard_with_bruteforce(self,debug_every,itr_combination,
                                                        output_area,total_combinations,
                                                        combination):
        if (debug_every > 0 
            and (
                (itr_combination+1) % debug_every == 0 
                or (itr_combination+1) == 1
                or (itr_combination+1) == total_combinations)
            ):
                with output_area:
                    output_area.clear_output(wait=True)
                    print('==========================================================')
                    print('ITERATION ',itr_combination+1,' OF ', total_combinations)
                    print('current combination: ',combination)
                    print()
                    print()
                    print()
                    print()
                    print('==========================================================\n')