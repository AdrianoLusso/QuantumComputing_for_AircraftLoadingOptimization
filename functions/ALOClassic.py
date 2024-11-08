
import itertools
import sys
import ipywidgets as widgets

from docplex.mp.model import Model

from openqaoa.problems.converters import FromDocplex2IsingModel


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time




class ALOClassic:    
    '''  
    This class implements the Aircraft Loading Optimization (ALO) and a couple of important methods for its classical
    representantion, calculus and solving.  
    
    FORMALISMS AND FORMATS CONVERSION 

        - standard form: 
            An optimization problem with objetive function and constraints functions. 
            It's expressed as a maximization problem.

        - QUBO form:
            a quadratic unconstrained optimization problem. As the formalism name says, it only accepts
            binary variables and the constraints are expressed as 'penalties' inside the objective function.
            It is expressed as a minimization problem, which would be later convenient for quantum algorithms.
            Each  binary variable x_{1*i+j} represents if the container 'i' is asigned to the plane position 'j'.

    
        - Ising form:
            makes the conversion of QUBO binary variables {0,1} to Ising binary variables {-1,1}. In this class, it 
            will be a QUBO object from OpenQAOA. While the name of the class could be confusing, when OpenQAOA talks
            about QUBO, it makes it about its Ising formulation.
    '''

    def __init__(self,instance_dict):
        '''
          Parameters:
            instance_dict:
                A dictionary with the instance configuration. 
                This dictionary should be previously created with ALORandomGenerator class,
                or by manually defining its elements.
            
          '''
        self.instance_dict = instance_dict
        self.mdl_qubo = None
        self.ising = None
          
        self.optimal_standard_solution = None
        self.optimal_standard_gain =  None
        self.optimized_lambdas = None
          


    ''' REPRESENTATIONS '''
    def to_qubo_and_ising(self,unbalanced=True,lambdas=[0.9603,0.0371],multipliers=100,debug=False):
        '''
        This method transforms the instance_dict of the ALO object into a:
            - docplex model with the QUBO formulation (no constraints, and just an objective function)
            - a QUBO object from OpenQAOA, which represents the Ising formulation.
        
        Parameters:
            - unbalanced
                a flag that marks if it will be used the typical slack variables approach or the unbalanced penalization
                approach for the inequalities constraints. 
                For the unbalanced approach, see https://iopscience.iop.org/article/10.1088/2058-9565/ad35e4.
            - lambdas
                the set of parameters for the unbalanced approach (won't be used if unbalanced = False)
                or the keyword 'optimized', which means that and custom optimized lambdas will be calculated
            -multipliers
                a number, or list of numbers, with the lagrange multipliers for the penalties that are not from the
                unbalanced approach
        Return:
            - the QUBO docplex model
            - the Ising formulation, as a OpenQAOA's QUBO object
        '''

        # make global variables from the instance_dict items
        for key, value in self.instance_dict.items():
            globals()[key] = value
        containers = range(num_containers)
        positions = range(num_positions)
        
        # create the docplex model
        mdl = Model("ALO")
        x = mdl.binary_var_list(len(allBinaryVariables), name="x")
        
        # OBJECTIVE FUNCTION
        # TODO MAYBE FIND A WAY TO MAKE X A 2-DIM ARRAY? IMPROVE READABILITY
        # TODO PROBAR DE NUEVO EL EXPERIMENTO PERO CON 10 VARIABLES Y P=3
        mdl.minimize(
            -mdl.sum(
                t_i * m_i * x[num_positions * i + j] 
                for j in positions
                for t_i,m_i,i in zip(t_list,containers_weight,containers)
            )
            +40* # CONTIGUITY OF BIG CONTAINERS --- P_C=40 is the penalty value for this penalty.
            mdl.sum(
                0.5 * mdl.sum(
                        x[len(positions) * i + j]
                        for j in positions
                    )
                -
                mdl.sum(
                    x[len(positions) * i + j] * x[len(positions) * i + (j+1)]
                    for j in positions[:-1]
                )
                for i in containers
                if containers_type[i]==3
            )
        )

        # add some linear penalties depending on the approach taken
        if unbalanced:
            self.__unbalanced_penalization_approach(mdl,x,containers,positions)
        else:
            self.__slack_variables_approach(mdl,x,containers,positions)
 
        # MAXIMUM CAPACITY --- its penalty value will be P_W
        mdl.add_constraint(
            mdl.sum(
                t_i * m_i * x[len(positions) * i + j] 
                for j in positions
                for t_i,m_i,i in zip(t_list,containers_weight,containers)
            ) <= max_weight
            ,ctname="maximum capacity"
        )

        if debug:
            print('GENERAL OPTIMIZATION FORM:')
            print(mdl.objective_expr,'\n')
            print('constraints:')
            for c in mdl.iter_constraints():
                print(c)

        # define the converter from docplex to ising model
        # for more details on how the converter works, see https://github.com/entropicalabs/openqaoa/blob/main/src/openqaoa-core/openqaoa/problems/converters.py#L9
        converter = FromDocplex2IsingModel(mdl, unbalanced_const=unbalanced,strength_ineq=lambdas,multipliers=multipliers)

        # gets the ising and qubo docplex
        ising = converter.ising_model
        mdl_qubo = converter.qubo_docplex

        if debug:
            print('QUBO FORM:')
            print('linear terms:')
            for var in mdl_qubo.objective_expr.iter_terms():
                print(var)
            print()
            print('quad. terms:')
            for quad_term in mdl_qubo.objective_expr.iter_quads():
                print(quad_term)

        return ising,mdl_qubo,mdl
    
    def __unbalanced_penalization_approach(self,mdl,x,containers,positions):
        '''
        This method adds the 'no overlaps' and 'no duplicates' constraints for the unbalanced penalization
        approach of the ALO QUBO. These version are equalities constraints to 1.

        Parameters:
            - mdl
                the docplex model to use
            - x
                the set of docplex model's variables
            - containers
                the list of containers
            - positions
                the list of positions
        '''
        # NO OVERLAPS --- its penalty value will be P_O
        for j in positions:
            mdl.add_constraint(mdl.sum(
                d_i * x[len(positions) * i + j]
                for d_i,i in zip(d_list,containers) 
                ) == 1
                ,ctname="no overlaps"
            )

        # NO DUPLICATES --- its penalty value will be P_D. It will have to guarantee that P_D > 2*P_C
        for i in containers:
            t_i = t_list[i]
            mdl.add_constraint(t_i * mdl.sum(
                x[len(positions) * i + j]
                for j in positions 
                ) == 1
                ,ctname="no duplicates"
            )

    def __slack_variables_approach(self,mdl,x,containers,positions):
        '''
        This method adds the 'no overlaps' and 'no duplicates' constraints for the slack variables
        approach of the ALO QUBO. These versions are less or equal constraints to 1.

        Parameters:
            - mdl
                the docplex model to use
            - x
                the set of docplex model's variables
            - containers
                the list of containers
            - positions
                the list of positions
        '''
        
        # NO OVERLAPS --- its penalty value will be P_O
        for j in positions:
            mdl.add_constraint(mdl.sum(
                d_i * x[len(positions) * i + j]
                for d_i,i in zip(d_list,containers) 
                ) <= 1
                ,ctname="no overlaps"
            )

        # NO DUPLICATES --- its penalty value will be P_D. It will have to guarantee that P_D > 2*P_C
        for i in containers:
            t_i = t_list[i]
            mdl.add_constraint(t_i * mdl.sum(
                x[len(positions) * i + j]
                for j in positions 
                ) <= 1
                ,ctname="no duplicates"
            )

    def quboSolution_to_standardSolution(self,qubo_solution,check_feasibility = False,draw=False):
        '''
        This method transforms a given QUBO solution into the corresponding standard formulation solution.

        Parameters:
            - qubo_solution
            - check_feasibility
                if True, the feasibility of the given qubo solution will be checked before the transformation
            - draw
                if set to True, the final standard solution will be graphically drawn

        Return:
            the solution in the standard formulation
        '''

        if check_feasibility and not self.is_qubo_feasible(qubo_solution):
           raise ValueError('The qubo solution is not feasible')
        
        # the solution is divided in subsolutions, where each subsolution is a container and its M possible positions.
        n = self.instance_dict['num_containers']
        m = self.instance_dict['num_positions']
        subsolutions = [qubo_solution[i:i+m] for i in range(0, len(qubo_solution), m)]


        # an standard solution is a list of lists S. A sublist S[j] contains the containers in position j.
        # the basic standard solution is a list where all its sublists are empty. That means, that no container
        # is positioned yet.
        standard_solution = [[] for i in range(m)]
        
        # i go through each subsolution/container
        for container,subsolution in enumerate(subsolutions):
            # look the positions where the container is
            positions = [i for i, c in enumerate(subsolution) if c == '1'] 
            # i marked each poition ocuped by the container in the standard solution
            for p in positions:
                standard_solution[p].append(container)

        if draw:
            self.draw_standard_solution(standard_solution,n)
        return standard_solution

    def standardSolution_to_quboSolution(self,standard_solution,draw=False):
        '''
        '''
        if draw:
            self.draw_standard_solution(standard_solution,n)

        num_containers = self.instance_dict['num_containers']
        num_positions = self.instance_dict['num_positions']
        qubo_solution = ['0' for i in range(num_containers*num_positions)]
        for position_index,position in enumerate(standard_solution):
            for container in position:
                qubo_solution[container * num_positions + position_index] = '1'

        return "".join(qubo_solution)

    def draw_standard_solution(self, standard_solution,num_containers=10):
        '''
        This method takes a standard solution of an ALO problem and show it in a graph.

        Parameters:
            - standard_solution
            - num_containers
                for the creating of the graph, the number of container in the ALO instance is needed
        '''
        
        # decide the amount of colors depending on the amount of containers
        cmap = plt.get_cmap('hsv')
        colors = [cmap(i / num_containers) for i in range(num_containers)]

        max_circles_per_cell = max([len(position) for position in standard_solution]+[1])
        fig, ax = plt.subplots(figsize=(len(standard_solution), max_circles_per_cell / 2))
        ax.set_xlim(0, len(standard_solution))
        ax.set_ylim(0, max_circles_per_cell)
        
        # Draw the positions
        for i, position in enumerate(standard_solution):
            # draw the position rectangle
            ax.add_patch(patches.Rectangle((i, 0), 1, max_circles_per_cell, edgecolor='black', facecolor='white'))
            
            # Draw container numbers on positions
            for j, num in enumerate(position):
                # Draw a circle with the corresponding color to the container number
                color = colors[num % len(colors)] 
                circle = patches.Circle((i + 0.5, max_circles_per_cell - (j + 0.5)), 0.4, color=color)
                ax.add_patch(circle)
                ax.text(i + 0.5, max_circles_per_cell - (j + 0.5), str(num), color='white', ha='center', va='center', fontsize=12)
        
        # configuration
        ax.set_aspect('equal')
        ax.axis('off')

        plt.show()


    ''' CALCULATIONS '''
    
    def __calculate_optimized_lambdas(self,mdl):
        params = [0.9603,0.0371]
        lambdas_cost_function = self.__create_lambdas_cost_function(mdl)

        self.optimized_lambdas = minimize(lambdas_cost_function, params, method='Nelder-Mead')

        return self.optimized_lambdas

    def calculate_standard_gain(self,solution,minimization=False):
        ''' 
       this method calculate the gain of a solution for the standard ALO optimization function

        Parameters:
            - solution
                the solution given should be given in a standard formulation. That is:
                a list [p_0 , p_1 , p_2 , ... , p_n] where each variable p_j is a position represented by
                a list of integers values representing the containers. If a container c_i in is the sublist
                p_j it means that it is occupying the position j.

            - minimization (optional)
                if True, the standard ALO evaluated will be treated as a minimization problem. By default, is
                False becase the original ALO formulation is as a maximization problem. I case of minimization,
                the 'gain' should be mentioned as 'cost' 

        Return:
            - the gain of the solution given for the standard ALO
        '''        
        
        # veryfing that length of solution is the same as num_positions
        if(len(solution) != self.instance_dict['num_positions']):
            raise ValueError('The length of the solution must be equal to num_positions = ',self.instance_dict['num_positions'])

        t_list = self.instance_dict['t_list']
        containers_weight = self.instance_dict['containers_weight']

        # gain calculus
        gain = 0
        for j,position in enumerate(solution):
            for i in position:
                gain += t_list[i] * containers_weight[i]  

        # if its necessary, translate from maximization problem gain to minimization problem cost
        if minimization:
            gain *= -1
        return gain
    
    def is_qubo_feasible(self,solution):
        '''
        This method return if a qubo solution is feasible

        Parameters:
            - solution
            
        Return:
            True if the qubo is feasible, False if not
        '''
        solution = str(solution)

        if (
            self.violate_no_overlaps(solution) 
            or self.violate_no_duplication(solution)
            or self.violate_contiguity(solution)
            or self.violate_maximum_weight(solution)           
        ):
            return False
        else:
            return True


    ''' SOLVERS '''
    def solve_standard_with_bruteforce(self,debug_every=0,minimization=False,sleep=0):
        ''' 
        this method solve the standard (constrained) ALO optimization function through brute force

        Parameters:
            - debug_every (optional)
                an integer X that for 'printing a debug message every X combinations analyzed'. If 0,
                no message will be print. If 1, all combinations status will be shown
            minimization (optional)
                if True, the standard ALO evaluated will be treated as a minimization problem. By default, is
                False becase the original ALO formulation is as a maximization problem. I case of minimization,
                the 'gain' should be mentioned as 'cost' 

        Return:
            - the optimal solution and the optimal gain
        '''   
        output_area = None
        if debug_every != 0:
            output_area = widgets.Output()
            display(output_area)

        
        # getting instance data
        t_list = self.instance_dict['t_list']
        d_list = self.instance_dict['d_list']
        containers_weight = self.instance_dict['containers_weight']
        containers_type = self.instance_dict['containers_type']
        max_gain = self.instance_dict['max_weight']
        num_positions = self.instance_dict['num_positions']

        # calculate the search space -- it will be a bitstring where each bit shows if a container
        # is positioned in the plane. It DOES NOT gives information on which exact position occupies.
        combinations = itertools.product(range(2),repeat=self.instance_dict['num_containers'])
        total_combinations = 2**(self.instance_dict['num_containers'])

        opt_gain = None
        opt_combination = None
        for itr_combination,combination in enumerate(combinations):
            #print('current comb ',combination)
            indexes = [index for index,container in enumerate(combination) if container==1]
            
            # evaluate if all the containers marked have space inside the plane
            space_available = num_positions
            i = 0
            while i < len(indexes) and space_available >= 0:
                if containers_type[indexes[i]] == 1:
                    #print('sdsdsd')
                    space_available -= 1
                elif containers_type[indexes[i]] == 2:
                    space_available -= 0.5
                else:
                    space_available -= 2
                
                i += 1

            # if there isn't enough place, this combination is ignored
            if space_available < 0:
                self.__complete_debug_solve_standard_with_bruteforce1(debug_every,sleep,itr_combination,
                                                        minimization,output_area,total_combinations,
                                                        combination,'no place enough',opt_gain,opt_combination)
                #print('ERRO1')
                continue

            # if there current gain(weight) exceed the maximum gain(weight), the combination is ignored
            current_gain = sum(containers_weight[index] for index in  indexes)
            if current_gain > max_gain:
                #print('ERRO2')
                self.__complete_debug_solve_standard_with_bruteforce1(debug_every,sleep,itr_combination,
                                                        minimization,output_area,total_combinations,
                                                        combination,'too weight!',opt_gain,opt_combination)
                continue

            if (opt_gain is None 
                or (not minimization and opt_gain < current_gain)
                or (    minimization and opt_gain>current_gain)):
                #print('SUCCESS')
                opt_gain = current_gain
                opt_combination = combination

            # debug prints
            self.__complete_debug_solve_standard_with_bruteforce1(debug_every,sleep,itr_combination,
                                                        minimization,output_area,total_combinations,
                                                        combination,current_gain,opt_gain,opt_combination)
            
            if opt_gain == 100:
                break

        #print(opt_combination)
        # from the optimal combination found, we store Containers objects with their data.
        containers = [
                    Container(i,w,t,d) for i,(w,t,d) in 
                    enumerate(zip(containers_weight,containers_type,d_list))
                    if opt_combination[i] == 1
                    ]
        # the containers are sorted from the smaller to the larger one (containers_type).
        order = {2: 2, 1: 1, 3: 0}
        containers = sorted(containers, key=lambda x: order[getattr(x, 'type')])
        #print([c.d for c in containers])
        # preliminary optimal solution
        opt_solution = [[] for j in range(self.instance_dict['num_positions'])]

        if debug_every != 0:
            #output_area.close()
            output_area = widgets.Output()
            display(output_area)

        containers_assigned = []
        previous_type_3 = False
        for j,_ in enumerate(opt_solution):
            # debug prints
            self.__complete_debug_solve_standard_with_bruteforce2(debug_every,sleep,j,output_area,
                                                        self.instance_dict['num_positions'],opt_solution)
            if previous_type_3:
                previous_type_3 = False
                continue

            space_used = 0
            for container in containers: 
                    #print('position ',j,' , container ',container.id)
                    if container.id in containers_assigned:
                        continue

                    opt_solution[j].append(container.id)
                    containers_assigned.append(container.id)

                    if container.type == 3:
                        previous_type_3 = True
                        opt_solution[j+1].append(container.id)
                        break
                    space_used += container.d
                    #print(space_used)
                    if space_used >= 1:
                        #print('ERROR1')
                        break
        self.__complete_debug_solve_standard_with_bruteforce2(debug_every,sleep,j+1,output_area,
                                                        self.instance_dict['num_positions'],opt_solution)
        
        self.optimal_standard_solution = list(opt_solution)
        self.optimal_standard_gain =  opt_gain
        return list(opt_solution),opt_gain

    ''' AUXILIAR '''

    def __create_lambdas_cost_function(self,mdl):

        if self.optimal_standard_solution is None:
            self.solve_standard_with_bruteforce()

        q = QAOA()
        qiskit_device = create_device(location='local', name='vectorized')
        q.set_device(qiskit_device)
        q.set_circuit_properties(p=1, param_type='standard', init_type='ramp', mixer_hamiltonian='x')
        q.set_backend_properties(prepend_state=None, append_state=None)
        q.set_classical_optimizer(method='powell', maxfev=1000, tol=0.01,
                                optimization_progress=True, cost_progress=True, parameter_log=True)
        def cost(params): #FALTA MULTIPLIERS PARAM
            multipliers=100 # this is the penalty value for P_O, P_D and P_W
            
            converterWrapper = FromDocplex2IsingModel(mdl, unbalanced_const=True,strength_ineq=params,multipliers=multipliers)

            # gets the ising and qubo docplex
            ising = converterWrapper.ising_model
            
            # compile, brute force solve and get energy
            q.compile(ising)
            q.solve_brute_force(verbose=False)
            fake_opt_energy=q.brute_force_results['energy']

            opt_qubo_solution = self.standardSolution_to_quboSolution(self.optimal_standard_solution)
            opt_energy = bitstring_energy(q.cost_hamil,opt_qubo_solution)

            return (fake_opt_energy - opt_energy)**2

        return cost

    def violate_no_overlaps(self,solution):
        '''
        This method return if a qubo solution violates the 'no-overlaps' constraint.

        Parameters:
            - solution
            
        Return:
            True if the qubo is violated the constraint, False otherway.
        '''

        # the solution is divide in subsolutions, where each subsolution is a container and its J possible positions.
        n = self.instance_dict['num_positions']
        m = self.instance_dict['num_positions']
        subsolutions = [solution[i:i+n] for i in range(0, len(solution), n)]

        # we get the d_i values for evaluating the constraint
        d_list = self.instance_dict['d_list']

        sums = []
        # for each position j, we evaluate if the sumatory of d_i * p_ij <= 1.
        # Is for any position j the constraint is broken, it return false
        for j,_ in enumerate(range(m)):
            sums.append(0)
            
            for i,container_in_position in enumerate(subsolutions):
                # we add the value to the sumatory
                sums[j] += d_list[i] * int(container_in_position[j])
                
                # evaluate if the constraint has been broken
                if sums[j] > 1:
                    return True
        
        # the solution is feasible for this constraint
        return False
    
    def violate_no_duplication(self,solution):
        '''
        This method return if a qubo solution violates the 'no-duplication' constraint.

        Parameters:
            - solution
            
        Return:
            True if the qubo is violated the constraint, False otherway.
        '''

        # the solution is divide in subsolutions, where each subsolution is a container and its J possible positions.
        n = self.instance_dict['num_positions']
        subsolutions = [solution[i:i+n] for i in range(0, len(solution), n)]

        # we get the t_i values for evaluating the constraint
        t_list = self.instance_dict['t_list']

        # for each container i, we evaluate if the sumatory of t_i * p_ij <= 1.
        # Is for any container i the constraint is broken, it return false
        for i,container_in_position in enumerate(subsolutions):
            sum = 0
                
            for j,_ in enumerate(container_in_position):
                # we add the value to the sumatory
                sum += t_list[i] * int(container_in_position[j])
                    
                # evaluate if the constraint has been broken
                if sum > 1:
                    return True
            
        # the solution is feasible for this constraint
        return False
    
    def violate_contiguity(self,solution):
        '''
        This method return if a qubo solution violates the 'contiguit' constraint.

        Parameters:
            - solution
            
        Return:
            True if the qubo is violated the constraint, False otherway.
        '''

        # the solution is divide in subsolutions, where each subsolution is a container and its J possible positions.
        n = self.instance_dict['num_positions']
        subsolutions = [solution[i:i+n] for i in range(0, len(solution), n)]

        types = self.instance_dict['containers_type']

        # for each big container i, if a position is ocuppied, the next must be also ocuppied.
        for i,subsolution in enumerate(subsolutions):
            if types[i] != 3:
                continue

            j = 0
            while j < len(subsolution):
                if subsolution[j] == '1':
                    # evaluate if the constraint has been broken
                    if j+1 == len(subsolution) or subsolution[j+1] != '1':
                        return True
                    j = j + 2
                else:
                    j = j + 1

        # the solution is feasible for this constraint  
        return False

    def violate_maximum_weight(self,solution):
        '''
        This method return if a qubo solution violates the 'maximum weight' constraint.

        Parameters:
            - solution
            
        Return:
            True if the qubo is violated the constraint, False otherway.
        '''

        # gets the data from the instanse
        max_weight = self.instance_dict['max_weight']
        containers_weight = self.instance_dict['containers_weight']
        t_list = self.instance_dict['t_list']

        # the solution is divide in subsolutions, where each subsolution is a container and its J possible positions.
        n = self.instance_dict['num_positions']
        subsolutions = [solution[i:i+n] for i in range(0, len(solution), n)]

        # for each big container i and position i, evaluates it a weight should be added
        current_weight=0
        for i,subsolution in enumerate(subsolutions):
            for position in subsolution:
                    current_weight +=  int(position) * t_list[i] * containers_weight[i]

                    # evaluate if the constraint has been broken
                    if current_weight > max_weight:
                        return True
        
        # the solution is feasible for this constraint
        return False


    def __complete_debug_solve_standard_with_bruteforce1(self,debug_every,sleep,itr_combination,
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
                    print('1ST STEP: FIND THE CONTAINERS THAT MAXIMIZES THE WEIGHT AND FIT INTO DE AIRCRAFT')
                    print('==========================================================')
                    print('ITERATION ',itr_combination+1,' OF ', total_combinations)
                    print('current combination: ',combination)
                    print('current ',text,': ',gain)
                    print()
                    print('current optimal ',text,': ',opt_gain)
                    print('current optimal solution: ',opt_solution)
                    print('==========================================================\n')
                time.sleep(sleep) 

    
    def __complete_debug_solve_standard_with_bruteforce2(self,debug_every,sleep,itr_combination,output_area,
                                                        total_combinations,opt_solution):
        if (debug_every > 0 
            and (
                (itr_combination+1) % debug_every == 0 
                or (itr_combination+1) == 1
                or (itr_combination+1) == total_combinations)
            ):
                #if itr_combination != 0:
                #    self.__clear_previous_lines(8)
                with output_area:
                    output_area.clear_output(wait=True)
                    print('2ND STEP: ACCOMODATE THE CONTAINER IN THE AIRCRAFT POSITIONS')
                    print('==========================================================')
                    print('evaluating position ',itr_combination,'...')
                    print('current optimal solution: ',opt_solution)
                    print('==========================================================\n')
                time.sleep(sleep) 


class Container:
    '''
    This is an auxiliary class for representing a container.
    '''
    def __init__(self, id,weight,type, d):
        self.id = id
        self.weight = weight
        self.type = type
        self.d = d