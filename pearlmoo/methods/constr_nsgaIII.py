# -*- coding: utf-8 -*-
#"""
#@author: Paul
#  The following implementation of NSGA-III is adapted to NEORL and draw lots of inspiration
#  from DEAP implementation: https://github.com/DEAP/deap/blob/master/deap/tools/emo.py#L15 
#"""

import random
import numpy as np
from collections import defaultdict
import copy
import joblib
import pathos.multiprocessing

from neorl.evolu.crossover import cxES2point, cxESBlend
from neorl.evolu.discrete import encode_grid_to_discrete, decode_discrete_to_grid
from neorl.utils.seeding import set_neorl_seed

from neorl import ES
from itertools import chain
from pearlmoo.utils.tools import sortNondominated, sortLogNondominated, find_extreme_points, find_intercepts, associate_to_niche, niching, uniform_reference_points

# for multi-objective optimization
from pearlmoo.utils.tools import sortNondominated_constr, sortLogNondominated_constr, feasible

class constrNSGAIII(ES):
    """
    Parallel Fast Non-dominated Sorting Gentic Algorithm - III with constraints
    
    Only the seleciton operator differ from classical GA implementation. Hence, we choose create a subclass
    of ES implementation

    :param mode: (str) problem type, either ``min`` for minimization problem or ``max`` for maximization
    :param bounds: (dict) input parameter type and lower/upper bounds in dictionary form. Example: ``bounds={'x1': ['int', 1, 4], 'x2': ['float', 0.1, 0.8], 'x3': ['float', 2.2, 6.2]}``
    :param lambda\_: (int) total number of individuals in the population. Use Combination(M + p - 1, p) for nicely spaced points on the pareto front.
    :param cxmode: (str): the crossover mode, either 'cx2point' or 'blend'
    :param alpha: (float) Extent of the blending between [0,1], the blend crossover randomly selects a child in the range [x1-alpha(x2-x1), x2+alpha(x2-x1)] (Only used for cxmode='blend')
    :param cxpb: (float) population crossover probability between [0,1]
    :param mutpb: (float) population mutation probability between [0,1] 
    :param smin: (float): minimum bound for the strategy vector
    :param smax: (float): maximum bound for the strategy vector
    :param ncores: (int) number of parallel processors
    :param seed: (int) random seed for sampling
    
    NSGA-III specific parameters:

    :param sorting: (str) sorting type, ``standard`` or ``log``. The latter should be faster and is used as default.
    :param: p: (int) number of divisions along each objective for the reference points. The number of reference points is Combination(M + p - 1, p), where M is the number of objective
    :param ref_points: (list) of user inputs reference points. If none the reference points are generated uniformly on the hyperplane intersecting each axis at 1.
    :param paradigm: (str) optimization type, ``unconstrained`` or ``constrained``.
    """
    def __init__ (self, mode, bounds, fit, lambda_=60, mu = 60, cxmode='cx2point', 
                  alpha=0.5, cxpb=0.6, mutpb=0.3, smin=0.01, smax=0.5, clip=True, ncores=1, seed=None, p = 4,ref_points = None,sorting = 'log',paradigm="unconstrained",**kwargs):  
        
        set_neorl_seed(seed)
        super().__init__(mode = mode, bounds = bounds, fit = fit, lambda_=lambda_, mu=mu, cxmode=cxmode, 
                  alpha=alpha, cxpb=cxpb, mutpb=mutpb, smin=smin, smax=smax, clip=clip, ncores=ncores, seed=seed)
        # new hyper-parameters
        self.sorting = sorting
        self.paradigm = paradigm
        if self.paradigm not in ['constrained','unconstrained']:
            raise NotImplementedError('---error: Parameter paradigm must be either "constrained" or "unconstrained" not {}'.format(self.paradigm))
        
        #NSGA-III specific
        self.p = p
        self.ref_points = ref_points
        def fitness_wrapper(*args, **kwargs):
            fitness = fit(*args, **kwargs) 
            if self.paradigm == 'constrained':
                if isinstance(fitness,np.ndarray):
                    if mode == 'max':
                        pass
                    elif mode == 'min':
                        fitness['objectives'] = - fitness['objectives']
                elif isinstance(fitness['objectives'],list):
                    if mode == 'max':
                        fitness['objectives'] = np.array(fitness['objectives'])
                    elif mode == 'min':
                        fitness['objectives'] = -np.array(fitness['objectives'])
                else:
                    if mode == 'max':
                        fitness['objectives'] = np.array([fitness['objectives']])
                    elif mode == 'min':
                        fitness['objectives'] = -np.array([fitness['objectives']])
                return fitness
            elif self.paradigm == 'unconstrained':
                if isinstance(fitness,np.ndarray):
                    if mode == 'max':
                        return fitness
                    elif mode == 'min':
                        return -fitness
                elif isinstance(fitness,list):
                    if mode == 'max':
                        return np.array(fitness)
                    elif mode == 'min':
                        return -np.array(fitness)
                else:
                    if mode == 'max':
                        return np.array([fitness])
                    elif mode == 'min':
                        return -np.array([fitness])
        self.fit=fitness_wrapper   
    def fit_worker(self, x):
        #"""
        #Evaluates fitness of an individual.
        #"""
        
        
        #mir-grid
        if self.grid_flag:
            #decode the individual back to the int/float/grid mixed space
            x=decode_discrete_to_grid(x,self.orig_bounds,self.bounds_map) 
                    
        fitness = self.fit(np.array(x))
        return fitness
        
    def select(self,pop, k, ref_points, nd='standard', best_point=None,
             worst_point=None, extreme_points=None):
        """
        Implementation of NSGA-III selection

        :param pop: (dict) A list of pop to select from.
        :param k: (int) The number of pop to select.
        :param ref_points: (list) Reference points to use for niching.
        :param nd: (str) Specify the non-dominated algorithm to use: 'standard' or 'log'.
        :param best_point: (list) Best point found at previous generation. If not provided
            find the best point only from current pop.
        :param worst_point: (list) Worst point found at previous generation. If not provided
            find the worst point only from current pop.
        :param extreme_points: (list) Extreme points found at previous generation. If not provided
            find the extreme points only from current pop.
        :Returns best_dict: (dict) next population in dictionary structure
        
        """
        if self.paradigm == "constrained":
            if nd == "standard":
                pareto_fronts,flag = sortNondominated_constr(pop, k)
            elif nd == 'log':
                pareto_fronts,flag = sortLogNondominated_constr(pop, k)
            else:
                raise Exception("NSGA3: The choice of non-dominated sorting "
                            "method '{0}' is invalid.".format(nd))
            # Extract fitnesses as a np array in the nd-sort order
            # Use * -1 to tackle always as a minimization problem. Necessary here as well
            if flag: # only unfeasible element
                best_dict=defaultdict(list)
                #index=0
                index=pareto_fronts[0][0][0]
                best_dict[index] = pareto_fronts[0][0][1]
                for key in pareto_fronts[1][:k-1]:
                    #index+=1
                    index = key[0]
                    best_dict[index] = key[1]#pareto_fronts[0][0]#key[1]
                return best_dict#
            
            fitnesses = np.array([ind[1][2]['objectives'][0]  for f in pareto_fronts for ind in f if np.sum(ind[1][2]['constraints']) == 0])#
            fitnesses *= -1
            

        elif self.paradigm == "unconstrained":
            if nd == "standard":
                pareto_fronts = sortNondominated(pop, k)
            elif nd == 'log':
                pareto_fronts = sortLogNondominated(pop, k)
            else:
                raise Exception("NSGA3: The choice of non-dominated sorting "
                            "method '{0}' is invalid.".format(nd)) 
            # Extract fitnesses as a np array in the nd-sort order
            # Use * -1 to tackle always as a minimization problem. Necessary here as well
            fitnesses = np.array([ind[1][2] for f in pareto_fronts for ind in f])
            fitnesses *= -1
            # Get best and worst point of population, contrary to pymoo
            # we don't use memory
        

        if best_point is not None and worst_point is not None:
            best_point = np.min(np.concatenate((fitnesses, best_point), axis=0), axis=0)
            worst_point = np.max(np.concatenate((fitnesses, worst_point), axis=0), axis=0)
        else:
            best_point = np.min(fitnesses, axis=0)
            worst_point = np.max(fitnesses, axis=0)

        extreme_points = find_extreme_points(fitnesses, best_point, extreme_points)
        front_worst = np.max(fitnesses[:sum(len(f) for f in pareto_fronts), :], axis=0)
        intercepts = find_intercepts(extreme_points, best_point, worst_point, front_worst)
        niches, dist = associate_to_niche(fitnesses, ref_points, best_point, intercepts)

        # Get counts per niche for individuals in all front but the last
        niche_counts = np.zeros(len(ref_points), dtype=np.int64)
        index, counts = np.unique(niches[:-len(pareto_fronts[-1])], return_counts=True)
        niche_counts[index] = counts
        # Choose individuals from all fronts but the last
        chosen = list(chain(*pareto_fronts[:-1]))
        
        if self.paradigm == "unconstrained":# classic NSGAIII
            # Use niching to select the remaining individuals
            sel_count = len(chosen)
            n = k - sel_count
            selected = niching(pareto_fronts[-1], n, niches[sel_count:], dist[sel_count:], niche_counts)
            chosen.extend(selected)
        elif self.paradigm == "constrained":
            sel_count = len(chosen)
            feasible_index = 0
            for count,pareto in enumerate(pareto_fronts[-1]):
                if np.sum(pareto[1][2]["constraints"]) == 0:
                    feasible_index += 1
                else:
                    break
            if feasible_index <= 1:# either the first or the second one
                selected = pareto_fronts[-1][feasible_index - 1]
                chosen.extend([tuple(selected)])
        
            else:
                #n = k - sel_count # only select ones that are feasible with niching, the rest is ranked by CV value
                if len(pareto_fronts[-1][:feasible_index]) < k:
                    n = len(pareto_fronts[-1][:feasible_index])
                else:
                    n = k
                selected = niching(pareto_fronts[-1][:feasible_index], n, niches[sel_count:sel_count + len(pareto_fronts[-1][:feasible_index])], dist[sel_count:sel_count + len(pareto_fronts[-1][:feasible_index])], niche_counts) 
                chosen.extend(selected)

            for fronts in pareto_fronts[-1][feasible_index:]:# --- the last pareto front has too many samples. First order the current by front and sorting distances
                sel_count = len(chosen)
                if k - sel_count == 0:
                    break
                chosen.extend([tuple(fronts)])
                
        # re-cast into a dictionary to comply with NEORL 
        best_dict=defaultdict(list)
        index=0
        for key in chosen:
            best_dict[index] = key[1]
            index+=1
        
        return best_dict
    def GenOffspring(self, pop):
        #"""
        # 
        #This function generates the offspring by applying crossover, mutation **or** reproduction. 
        #The sum of both probabilities self.cxpb and self.mutpb must be in [0,1]
        #The reproduction probability is 1 - cxpb - mutpb
        #The new offspring goes for fitness evaluation
        
        #Inputs:
        #    pop (dict): population in dictionary structure
        #Returns:
        #    offspring (dict): new modified population in dictionary structure    
        #"""
        
        
        pop_indices=list(pop.keys())#
        offspring = defaultdict(list)
        for i in range(self.lambda_):
            rn = random.random()
            #------------------------------
            # Crossover
            #------------------------------
            if rn < self.cxpb:        
                if self.paradigm == "constrained":
                    # Tournament to select two parents
                    # first parent:
                    tempindex1, tempindex2 = random.sample(pop_indices,2)
                    E_1,E_2 = pop[tempindex1][2],pop[tempindex2][2]
                    if not feasible(E_1) and feasible(E_2):
                        index1 = tempindex2
                    elif feasible(E_1) and not feasible(E_2):
                        index1 = tempindex1
                    elif not feasible(E_1) and not feasible(E_2):
                        if np.sum(E_1["constraints"]) >= np.sum(E_2["constraints"]):
                            index1 = tempindex2
                        else:
                            index1 = tempindex1
                    elif feasible(E_1) and feasible(E_2):
                        index1 = random.choice([tempindex1,tempindex2])
                    # second parent:
                    tempindex1, tempindex2 = random.sample(pop_indices,2)
                    E_1,E_2 = pop[tempindex1][2],pop[tempindex2][2]
                    if not feasible(E_1) and feasible(E_2):
                        index2 = tempindex2
                    elif feasible(E_1) and not feasible(E_2):
                        index2 = tempindex1
                    elif not feasible(E_1) and not feasible(E_2):
                        if np.sum(E_1["constraints"]) > np.sum(E_2["constraints"]):
                            index2 = tempindex2
                        else:
                            index2 = tempindex1
                    elif feasible(E_1) and feasible(E_2):
                        index2 = random.choice([tempindex1,tempindex2])
                    
                elif self.paradigm == "unconstrained":    
                    index1, index2 = random.sample(pop_indices,2)
                    E_1,E_2 = pop[index1][2],pop[index2][2]
                
                if self.cxmode.strip() =='cx2point':
                    ind1, ind2, strat1, strat2 = cxES2point(ind1=list(pop[index1][0]),ind2=list(pop[index2][0]), 
                                                            strat1=list(pop[index1][1]),strat2=list(pop[index2][1]))
                elif self.cxmode.strip() == 'blend':
                    ind1, ind2, strat1, strat2=cxESBlend(ind1=list(pop[index1][0]), ind2=list(pop[index2][0]), 
                                                                         strat1=list(pop[index1][1]),strat2=list(pop[index2][1]),
                                                                         alpha=self.alpha)
                else:
                    raise ValueError('--error: the cxmode selected (`{}`) is not available in ES, either choose `cx2point` or `blend`'.format(self.cxmode))
                
                ind1=self.ensure_bounds(ind1)
                ind2=self.ensure_bounds(ind2)
                
                ind1=self.ensure_discrete(ind1)  #check discrete variables after crossover
                ind2=self.ensure_discrete(ind2)  #check discrete variables after crossover
                
                offspring[i + len(pop)].append(ind1)
                offspring[i + len(pop)].append(strat1)
            
            #------------------------------
            # Mutation
            #------------------------------
            elif rn < self.cxpb + self.mutpb:  # Apply mutation
                index = random.choice(pop_indices)
                ind, strat=self.mutES(ind=list(pop[index][0]), strat=list(pop[index][1]))
                offspring[i + len(pop)].append(ind)
                offspring[i + len(pop)].append(strat)
            
            #------------------------------
            # Reproduction from population
            #------------------------------
            else:                         
                index=random.choice(pop_indices)
                pop[index][0]=self.ensure_discrete(pop[index][0])
                offspring[i + len(pop)].append(pop[index][0])
                offspring[i + len(pop)].append(pop[index][1])
                
        if self.clip:
            for item in offspring:
                offspring[item][1]=list(np.clip(offspring[item][1], self.smin, self.smax))
        
        return offspring
               
    def evolute(self, ngen, x0=None, verbose=False):
        """
        This function evolutes the NSGA-III algorithm for number of generations.
        
        :param ngen: (int) number of generations to evolute
        :param x0: (list of lists) the initial position of the swarm particles
        :param verbose: (bool) print statistics to screen
        
        :return: (tuple) (best individual, best fitness, and a list of fitness history)
        """
        self.es_hist={}
        self.es_hist['mean_strategy']=[]
        self.best_scores=[]
        self.best_indvs=[]
        if x0:    
            assert len(x0) == self.lambda_, '--error: the length of x0 ({}) (initial population) must equal to the size of lambda ({})'.format(len(x0), self.lambda_)
            self.population=self.init_pop(x0=x0, verbose=verbose)
        else:
            self.population=self.init_pop(verbose=verbose)
            
        if self.paradigm == 'constrained':
            self.y_opt=[-np.inf for i in range(len(self.population[1][2]['objectives'][0]))]
            self.x_opt=[[] for i in range(len(self.population[1][2]['objectives'][0]))]
        elif self.paradigm == 'unconstrained':
            self.y_opt=[-np.inf for i in range(len(self.population[1][2]))]
            self.x_opt=[[] for i in range(len(self.population[1][2]))]
        if len(self.population[1][2]) == 1:
            print("--warning: length of output is 1, the sorting method is changed to ``standard``.")
            self.sorting = "standard"
        # generate reference points
        if self.ref_points is None:
            if self.paradigm == "constrained":
                self.ref_points = uniform_reference_points(nobj = len(self.population[1][2]['objectives'][0]), p = self.p) 
            elif self.paradigm == "unconstrained":
                self.ref_points = uniform_reference_points(nobj = len(self.population[1][2]), p = self.p)   
        # Begin the evolution process
        for gen in range(1, ngen + 1):
            
            # Vary the population and generate new offspring
            offspring = self.GenOffspring(pop=self.population)
            # Evaluate the individuals with an invalid fitness with multiprocessign Pool
            # create and run the Pool
            if self.ncores > 1:
                core_list=[]
                for key in offspring:
                    core_list.append(offspring[key][0])

                try:
                    with joblib.Parallel(n_jobs=self.ncores) as parallel:
                        fitness=parallel(joblib.delayed(self.fit_worker)(item) for item in core_list)
                except:
                    p=pathos.multiprocessing.Pool(processes = self.ncores)
                    fitness = p.map(self.fit_worker, core_list)
                    p.close()
                    p.join()
                    
                for ind in range(len(offspring)):
                    offspring[ind + len(self.population)].append(fitness[ind]) 
                
            else: #serial calcs
                
                for ind in offspring.keys():
                    fitness=self.fit_worker(offspring[ind][0])
                    offspring[ind].append(fitness)
        
                
            # Select the next generation population
            offspring.update(self.population) # concatenate offspring and parents dictionnaries
            self.population = copy.deepcopy(self.select(pop=offspring, k=self.mu, ref_points = self.ref_points, nd = self.sorting))
            if self.RLmode:  #perform RL informed ES
                self.population=self.mix_population(self.population)
            
            if self.paradigm == "constrained":
                if self.sorting == "standard":
                    pareto_front,flag = sortNondominated_constr(self.population, len(self.population))
                elif self.sorting == 'log':
                    pareto_front,flag = sortLogNondominated_constr(self.population, len(self.population))
                pareto_front = pareto_front[0]
                
                inds_par, rwd_par=[i[1][0] for i in pareto_front], [i[1][2]['objectives'][0] for i in pareto_front]#
                rwd_par_constr=[i[1][2]['constraints'] for i in pareto_front]
                
                if self.mode=='max':
                    self.best_scores.append(np.concatenate((rwd_par,rwd_par_constr),axis=1))
                else:
                    self.best_scores.append(np.concatenate(([-x for x in rwd_par],rwd_par_constr),axis=1))
            elif self.paradigm == "unconstrained":
                if self.sorting == "standard":
                    pareto_front = sortNondominated(self.population, len(self.population))[0]
                elif self.sorting == 'log':
                    pareto_front = sortLogNondominated(self.population, len(self.population))[0]   
                inds_par, rwd_par=[i[1][0] for i in pareto_front], [i[1][2] for i in pareto_front]
                if self.mode=='max':
                    self.best_scores.append(rwd_par)
                else:
                    self.best_scores.append([-x for x in rwd_par])
            
                inds_par, rwd_par=[i[1][0] for i in pareto_front], [i[1][2] for i in pareto_front]
            
            if self.mode=='max':
                self.es_hist['utopia'] = list(map(max,zip(*rwd_par)))
                self.es_hist['nadir'] = list(map(min,zip(*rwd_par)))
            else:
                self.es_hist['utopia'] = list(map(min,zip(*[-x for x in rwd_par])))
                self.es_hist['nadir'] = list(map(max,zip(*[-x for x in rwd_par])))
            if self.grid_flag:
                temp_indvs = []
                for count,elem in enumerate(inds_par):
                    temp_indvs.append(decode_discrete_to_grid(elem,self.orig_bounds,self.bounds_map))
                self.best_indvs.append(temp_indvs)
            else:
                self.best_indvs.append(inds_par)
            for fitn in range(len(np.min(rwd_par,axis=0))):
                if self.mode == 'min':
                    if  - np.min(rwd_par,axis=0)[fitn] > self.y_opt[fitn]:
                        self.y_opt[fitn] = np.min(rwd_par,axis=0)[fitn]
                        self.x_opt[fitn]=copy.deepcopy(inds_par[np.argmin(rwd_par,axis=0)[fitn]])
                elif self.mode == 'max':
                    if  np.min(rwd_par,axis=0)[fitn] > self.y_opt[fitn]:
                        self.y_opt[fitn] = np.min(rwd_par,axis=0)[fitn]
                        self.x_opt[fitn]=copy.deepcopy(inds_par[np.argmin(rwd_par,axis=0)[fitn]])
            #--mir
            if self.mode=='min':
                self.y_opt_correct=[- x for x in self.y_opt]
            else:
                self.y_opt_correct=self.y_opt

            #mir-grid

            if self.grid_flag:
                self.x_opt_correct = []
                for count,elem in enumerate([inds_par[x] for x in np.argmin(rwd_par,axis=0)]):
                    self.x_opt_correct.append(decode_discrete_to_grid(elem,self.orig_bounds,self.bounds_map))#self>x_opt
            else:
                self.x_opt_correct=[inds_par[x] for x in np.argmin(rwd_par,axis=0)]#inds_par#self.x_opt
            
            mean_strategy=[np.mean(self.population[i][1]) for i in self.population]
            self.es_hist['mean_strategy'].append(np.mean(mean_strategy))
            if verbose:
                print('##############################################################################')
                print('NSGA-III step {}/{}, CX={}, MUT={}, MU={}, LAMBDA={}, Ncores={}'.format(gen*self.lambda_,ngen*self.lambda_, np.round(self.cxpb,2), np.round(self.mutpb,2), self.mu, self.lambda_, self.ncores))
                print('##############################################################################')
                print('Statistics for generation {}'.format(gen))
                print('Best Fitness:', np.min(rwd_par,axis=0) if self.mode == 'max' else -np.min(rwd_par,axis=0))
                print('Best Individual(s):', self.x_opt_correct)
                print('Length of the pareto front / length of the population: {} / {}'.format(len(inds_par),len(self.population)))
                print('Max Strategy:', np.round(np.max(mean_strategy),3))
                print('Min Strategy:', np.round(np.min(mean_strategy),3))
                print('Average Strategy:', np.round(np.mean(mean_strategy),3))
                print('##############################################################################')
        if verbose:
            print('------------------------ NSGA-III Summary --------------------------')
            print('Best fitness (y) found:', self.y_opt_correct)
            print('Best individual (x) found:', self.x_opt_correct)
            print('--------------------------------------------------------------') 

        
        #---update final logger
        
        if self.paradigm == "constrained":
            if self.mode == 'min':
                self.best_scores=[-np.array(item[:-1]) for item in self.best_scores]
                self.es_hist['global_fitness'] = np.concatenate(([-x for x in rwd_par],rwd_par_constr),axis=1)
            else:
                self.es_hist['global_fitness'] = np.concatenate((rwd_par,rwd_par_constr),axis=1) # pareto of last population

        if self.paradigm == "unconstrained":
            if self.mode == 'min':
                self.best_scores=[-np.array(item) for item in self.best_scores]
                self.es_hist['global_fitness'] = [- x for x in rwd_par]
            else:
                self.es_hist['global_fitness'] = rwd_par # pareto of last population
        self.es_hist['global_pop'] = inds_par
        
        self.es_hist['local_fitness'] = self.best_scores # full history of pareto front
        self.es_hist['local_pop'] = self.best_indvs
        
        return self.x_opt_correct, self.y_opt_correct, self.es_hist
    