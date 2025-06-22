# -*- coding: utf-8 -*-
#"""#
#@author: Paul seurin
#    Thanks to https://github.com/aims-umich/neorl, this script is modified or adapted from this repo 
#"""

import sys, uuid
from neorl.evolu.discrete import encode_grid_to_discrete, decode_discrete_to_grid
from neorl.rl.baselines.shared import set_global_seeds
from neorl.rl.baselines.shared.vec_env import SubprocVecEnv
import numpy as np
import gym
from gym.spaces import Box, MultiDiscrete, Discrete
import random


import copy,sys
from scipy.special import rel_entr
import pandas as pd
import time,os

def globalize(func):
   """
   multiprocessing trick to paralllelize nested functions in python 
   (un-picklable objects!)
   """
   def result(*args, **kwargs):
       return -func(*args, **kwargs)
   result.__name__ = result.__qualname__ = uuid.uuid4().hex
   setattr(sys.modules[result.__module__], result.__name__, result)
   return result

def action_map(norm_action, ub, lb, ub_norm, lb_norm):
    """
    Map a nromalized action `norm_action` from a small range [lb_norm, ub_norm] 
    to a new range [lb, ub].
    Ex: Convert norm_action from [-1, 1] to [-100, 100]
    """
    d=len(norm_action)
    NormRange = np.array([ub_norm-lb_norm]*d)
    OrigRange = ub - lb
    NormMin=np.array([lb_norm]*d)  
    OrigMin = lb
    new_action = ((norm_action - NormMin) * OrigRange / NormRange) + OrigMin
    
    return new_action
            
def ensure_discrete(action, var_type):
    """
    Check the variables in a vector `vec` and convert the discrete ones to integer
    based on their type saved in `var_type`
    """
    vec=[]
    for dim in range(len(action)):
        if var_type[dim] == 'int':
            vec.append(int(action[dim]))
        else:
            vec.append(action[dim])
    return vec

def convert_actions_multidiscrete(bounds):
    """
    For PPO/ACKTR/A2C, convert the action provided by the user to a multidiscrete vector
    to be compatible with OpenAI gym multidiscrete space.
    Input: Provide the bounds dict for all variables.
    Returns: action_bounds (list) for encoding and bounds_map (dict) for decoding
    """
    action_bounds=[]
    
    for item in bounds:
        action_bounds.append(len(list(range(bounds[item][1],bounds[item][2]+1))))
    
    bounds_map={}
    for var, act in zip(bounds,action_bounds):
        bounds_map[var]={}
        act_list=list(range(bounds[var][1],bounds[var][2]+1))
        for i in range(act):
            bounds_map[var][i] = act_list[i]
    
    return action_bounds, bounds_map 

def convert_multidiscrete_actions(action, int_bounds_map):
    """
    For PPO/ACKTR/A2C, convert the action in multidiscrete form 
    to the real action space defined by the user
    Input: Provide the action in multidiscrete, and the integer bounds map
    Returns: decoded action (list)
    """
    decoded_action=[]
    
    for act, key in zip(action, int_bounds_map):
        decoded_action.append(int_bounds_map[key][act])
        
    return decoded_action 

class BaseEnvironment(gym.Env):
    #"""
    #A module to construct a fitness environment for certain algorithms 
    #that follow reinforcement learning approach of optimization
    #
    #:param method: (str) the supported algorithms, choose either: ``ppo``, ``acktr``, ``a2c``
    #:param fit: (function) the fitness function
    #:param bounds: (dict) input parameter type and lower/upper bounds in dictionary form. Example: ``bounds={'x1': ['int', 1, 4], 'x2': ['float', 0.1, 0.8], 'x3': ['float', 2.2, 6.2]}``
    #:param mode: (str) problem type, either ``min`` for minimization problem or ``max`` for maximization (RL is default to ``max``)
    # PEARL specific entries:
    #:param selection: (str) type of sub-routine in PEARL to assign the rank of the solutions generated compared to the best solutions stored.
    #:param sorting: (str) sorting type, ``standard`` or ``log``. The latter should be faster and is used as default.
    #:param buffer_size: (int) number of solutions stored in the memory of each agent. The lowest rank assigned to a solution generated is therefore equal to buffer_size.
    #:param: p: (int) number of divisions along each objective for the reference points. The number of reference points is Combination(M + p - 1, p), where M is the number of objective
    #:param ref_points: (list) of user inputs reference points. If none the reference points are generated uniformly on the hyperplane intersecting each axis at 1. Utilised when niching and niching2 are used for selection.
    #:param buffer_pop: (None or csv) utilized to store solutions and rank new ones. If a csv, the agent will start with old solutions generated, giving a head start to the algorithm in terms of which solutions are good.
    #:param paradigm: (str) type of optimization, whether constrained or unconstrained.
    #:param cl_penalty: (float) fixed constraints assigned to the reward of non-feasible solutions. We recommand it to be equal to buffer_size.
    #:param nadir: (list) nadir (theoretical worst solution) used to calculate the hypervolume of the population of solutions found. Utilized in the callback.
    #"""
    def __init__(self, method, fit, bounds, mode='max', 
        selection="crowding",sorting="log",buffer_size = 100, p = 4,ref_points = None,buffer_pop=None,
        paradigm="unconstrained",cl_penalty=1000.0 , nadir=None):
        if method not in ['ppo', 'a2c', 'acktr']:
            raise ValueError ('--error: unknown RL method is provided, choose from: ppo, a2c, acktr')
        self.episode_length=1# Always equal to one. See the paper referenced from PEARL.
        self.var_type = np.array([bounds[item][0] for item in bounds])
        self.nx=len(bounds)
        self.method=method
        
        # Dictionary for statistics and post-processing
        self.pearl_hist = {}
        self.pearl_hist['local_fitness'],self.pearl_hist['local_solution'],self.pearl_hist['objective'] = [],[],[]
        self.nadir = nadir
        #mir-grid
        if "grid" in self.var_type:
            self.grid_flag=True
            self.orig_bounds=bounds  #keep original bounds for decoding
            print('--debug: grid parameter type is found in the space')
            self.bounds, self.bounds_map=encode_grid_to_discrete(bounds) #encoding grid to int
            #define var_types again by converting grid to int
            self.var_type = np.array([self.bounds[item][0] for item in self.bounds])
        else:
            self.grid_flag=False
            self.bounds = bounds

        self.lb=np.array([self.bounds[item][1] for item in self.bounds])
        self.ub=np.array([self.bounds[item][2] for item in self.bounds])
        
        #for PPO/A2C/ACKTR gaussian policy, keep action range normalized, then map it back
        #Recommended---> -1/1 or -2/2        
        self.lb_norm=-1
        self.ub_norm=1
        #--mir
        self.mode=mode
        if mode == 'max':
            self.fit=fit
        elif mode == 'min':
            self.fit = globalize(lambda x: fit(x))  #use the function globalize to serialize the nested fit
        else:
            raise ValueError('--error: The mode entered by user is invalid, use either `min` or `max`')
        
        
         
        self.multi_objective_selection = selection
        assert selection in ['niching','crowding','niching2','crowding2'], '--error: The selection entered by the user is invalid. It must be either niching, crowding, niching2, or crowding2 not {}'.format(selection)
        
        self.paradigm=paradigm
        if self.paradigm == 'uncsontrained' and selection in ['niching2','crowding2']:
            raise ValueError('--error: niching2 or crowding2 are methods applied only to unconstrained cases. Consider niching or crowding instead.')
        
        self.p = p
        
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

        if self.paradigm == "constrained":
            self.best_constr = np.float('inf')
            self.cl_penalty = cl_penalty
        if selection not in ['crowding2','niching2']: # Constraints are first solved then the optimization problem in the feasibility space is solved
            self.constr_option='CL'# Curriculum Learning, for niching and crowding in selection. Only option allowed.
        else:
            self.constr_option = 'BASE'# The penalty when a solution is not feasible is utilized in the ranking 
        if self.paradigm not in ['constrained','unconstrained']:
            raise NotImplementedError('--error: Parameter paradigm must be either "constrained" or "unconstrained" not {}'.format(self.paradigm))
        
        if all([item == 'int' for item in self.var_type]):   #discrete optimization
            if method in ['ppo', 'a2c', 'acktr']:
                self.action_bounds, self.int_bounds_map=convert_actions_multidiscrete(self.bounds)
                self.action_space = MultiDiscrete(self.action_bounds)
                
                self.multi_objective_size = buffer_size
                self.multi_objective_sorting = sorting
                self._episode = 0
                # generate reference points
                if ref_points is None:
                    if self.paradigm == "constrained":
                        size_ref_points = len(self.fit(self.action_space.sample())["objectives"][0])
                            
                    elif self.paradigm == "unconstrained":
                        size_ref_points = len(self.fit(self.action_space.sample()))
                    self.multi_objective_ref_points = uniform_reference_points(nobj = size_ref_points, p = self.p) 
                else:
                    self.multi_objective_ref_points = ref_points
                    size_ref_points = len(ref_points)
                
                self.observation_space = Box(low=self.lb, high=self.ub)
                
                if buffer_pop is None:
                    self.buffer_pop = {}
                else:
                    self.buffer_pop = {}
                    
                    data = pd.read_csv(buffer_pop).values
                    
                    if self.paradigm == "constrained":
                        for row in range(data.shape[0]):
                            if self.multi_objective_selection in ["niching2","crowding2"]:
                                self.buffer_pop[row + 1] = [self.observation_space.sample().tolist(),{"objectives":list(data[row,1:self.n_obj+1]),"constraints":np.array(list(data[row,self.n_obj+1:]))},np.array(list(data[row,self.n_obj+1:]))]
                            else:
                                self.buffer_pop[row + 1] = [self.observation_space.sample().tolist(),list(data[row,1:self.n_obj+1]),np.array(list(data[row,self.n_obj+1:]))]
                            
                    else:
                        for row in range(data.shape[0]):
                            self.buffer_pop[row + 1] = [self.observation_space.sample().tolist(),list(data[row,1:-1])]
                self.cont_map_flag=False
                self.int_map_flag=True
        else:
            if method in ['ppo', 'a2c', 'acktr']:
                self.action_space = Box(low=self.lb_norm, high=self.ub_norm, shape=(self.nx,))
                self.multi_objective_size = buffer_size
                self.multi_objective_sorting = sorting
                self._episode = 0
                # generate reference points
                if ref_points is None:
                    if self.paradigm == "constrained":
                        size_ref_points = len(self.fit(self.action_space.sample())["objectives"][0])
                    elif self.paradigm == "unconstrained":
                        size_ref_points = len(self.fit(self.action_space.sample()))
                    self.multi_objective_ref_points = uniform_reference_points(nobj = size_ref_points, p = self.p) 
                else:
                    self.multi_objective_ref_points = ref_points
                    size_ref_points = len(ref_points)
                self.n_obj = size_ref_points
                
                self.observation_space = Box(low=self.lb, high=self.ub)
                
                
                if buffer_pop is None:
                    self.buffer_pop = {}
                else:# Head start the buffer of solutions
                    self.buffer_pop = {}
                    data = pd.read_csv(buffer_pop).values#.dropna()
                    if self.paradigm == "constrained":
                        for row in range(data.shape[0]):
                            if self.multi_objective_selection in ["niching2","crowding2"]:
                                self.buffer_pop[row + 1] = [self.observation_space.sample().tolist(),{"objectives":list(data[row,1:self.n_obj+1]),"constraints":np.array(list(data[row,self.n_obj+1:]))},np.array(list(data[row,self.n_obj+1:]))]
                            else:
                                self.buffer_pop[row + 1] = [self.observation_space.sample().tolist(),list(data[row,1:self.n_obj+1]),np.array(list(data[row,self.n_obj+1:]))]       
                    else:
                        for row in range(data.shape[0]):
                            self.buffer_pop[row + 1] = [self.observation_space.sample().tolist(),list(data[row,1:-1])]
                self.cont_map_flag=True
                self.int_map_flag=False

        self.reset()
        self.done=False
        self.counter = 0
        self.index=0

         
    

    def seed(self, seed_id):
        np.random.seed(seed_id)
        random.seed(seed_id)
        
    def step(self, action):
        state, action, reward=self.action_mapper(action)
        self.counter += 1

        # Record history of solutions found
        if self.paradigm == "constrained":
            if self.multi_objective_selection in ["niching2","crowding2"]:
                self.pearl_hist["global_fitness"] = [np.concatenate([x[1]["objectives"],x[1]["constraints"]]) for x in self.buffer_pop.values()]# best pareto front
            else:
                self.pearl_hist["global_fitness"] = [np.concatenate([x[1],x[2]]) for x in self.buffer_pop.values()]# best pareto front
        elif self.paradigm == "unconstrained":
            self.pearl_hist["global_fitness"] = [np.array(x[1]) for x in self.buffer_pop.values()]
        self.pearl_hist["local_solution"].append([x[0] for x in self.buffer_pop.values()])
        self.pearl_hist["global_solution"] = [x[0] for x in self.buffer_pop.values()]
        
        if self.counter == self.episode_length:# restart the episode
            self.done=True
            self.counter = 0
            self._episode += 1
        
        return state, reward, self.done, {'x':action}
    
    def reset(self):
        self.done=False
        if self.method in ['ppo', 'a2c', 'acktr']:
            init_state=self.action_space.sample()
        else:
            pass
        return init_state

    def render(self, mode='human'):
        pass
    
    def action_mapper(self, action):
        if self.method in ['ppo', 'a2c', 'acktr']:
            #--------------------------
            # cont./discrete methods
            #---------------------------
            if self.cont_map_flag:
                action=action_map(norm_action=action, 
                                  lb=self.lb, ub=self.ub, 
                                  lb_norm=self.lb_norm, ub_norm=self.ub_norm)
                
            if self.int_map_flag:  #this flag converts multidiscrete action to the real space
                action=convert_multidiscrete_actions(action=action, int_bounds_map=self.int_bounds_map)
                
            if 'int' in self.var_type:
                action=ensure_discrete(action=action, var_type=self.var_type)
            action = self.ensure_bounds(action)   #checking the bounds
            
              
            if self.grid_flag:
                #decode the individual back to the int/float/grid mixed space
                decoded_action=decode_discrete_to_grid(action,self.orig_bounds,self.bounds_map)
                list_of_obj=self.fit(decoded_action)  #calc reward based on decoded action
                state=action.copy()   #save the state as the original undecoded action (for further procecessing)
                action=decoded_action  #now for logging, return the action as the decoded action
            else:
                #calculate reward and use state as action
                list_of_obj=self.fit(action)  
                state=action.copy()     
            
            # Add the objective to see the evolution of each objective at each step (refer to the paper for understanding its importance)
            self.pearl_hist['objective'].append(list_of_obj)
            if self.paradigm == "constrained":
                constraints_obj = np.abs(list_of_obj["constraints"])
                list_of_obj = list_of_obj["objectives"][0]
                if self.best_constr > np.sum(constraints_obj):
                    self.best_constr = np.sum(constraints_obj)
                if self.constr_option == "CL" and np.sum(constraints_obj) != 0:
                    reward = - (np.sum(constraints_obj)) - self.cl_penalty
                    return state, action, reward
                 
                
            if self.multi_objective_selection in ["niching2","crowding2"]:
                if list_of_obj.tolist() not in list(map(lambda x: x[1]["objectives"],self.buffer_pop.values())):
                   self.buffer_pop[len(self.buffer_pop.keys()) + 1] = [state.tolist(),{"objectives":list_of_obj.tolist(),"constraints":np.array(constraints_obj)},constraints_obj.tolist()]    
            else:    
                if list_of_obj.tolist() not in list(map(lambda x: x[1],self.buffer_pop.values())):
                    if self.paradigm == "constrained":
                        self.buffer_pop[len(self.buffer_pop.keys()) + 1] = [state.tolist(),list_of_obj.tolist(),constraints_obj.tolist()]
                    elif self.paradigm == "unconstrained":
                        self.buffer_pop[len(self.buffer_pop.keys()) + 1] = [state.tolist(),list_of_obj.tolist()]
            
            chosen = copy.deepcopy(select(pop=self.buffer_pop, k=self.multi_objective_size, selection=self.multi_objective_selection,nd = self.multi_objective_sorting,ref_points=self.multi_objective_ref_points))
            position = [x[0] for x in chosen]
            if len(self.buffer_pop.keys()) in position:
                reward = - position.index(len(self.buffer_pop.keys()))
            else:
                reward = - self.multi_objective_size
            self.buffer_pop = dict([(count + 1,attr[1]) for count,attr in enumerate(chosen) if count <= self.multi_objective_size])# if count + 1 <= self.multi_objective_size]) # update the global buffer

        
        #if self.multi_objective_selection != "niching2" and self.multi_objective_selection != "crowding2":
        #    if self.paradigm == "constrained": # re-rank with feasibility
        #        if self.multi_objective_sorting == 'standard':
        #            pareto_fronts,flag = sortNondominated_constrX(self.buffer_pop, self.multi_objective_size)
        #        elif self.multi_objective_sorting == "log":
        #            pareto_fronts,flag = sortLogNondominated_constrX(self.buffer_pop, self.multi_objective_size)
        #        chosen = []
        #        for fronts in pareto_fronts:
        #            chosen.extend(list([fronts]))
        #        
        #        self.buffer_pop = dict([(count + 1,attr[1]) for count,attr in enumerate(chosen[0])])
                
        return state, action, reward

    def ensure_bounds(self, vec): # bounds check
    
        vec_new = []
    
        for i, (key, val) in enumerate(self.bounds.items()):
            # less than minimum 
            if vec[i] < self.bounds[key][1]:
                vec_new.append(self.bounds[key][1])
            # more than maximum
            if vec[i] > self.bounds[key][2]:
                vec_new.append(self.bounds[key][2])
            # fine
            if self.bounds[key][1] <= vec[i] <= self.bounds[key][2]:
                vec_new.append(vec[i])
        
        return np.array(vec_new)            
    

def CreateEnvironment(method, fit, bounds, ncores=1, p = 4, mode='max', selection="crowding",sorting="log",buffer_size = 100,ref_points = None,paradigm="unconstrained",cl_penalty=1000,buffer_pop=None,nadir=None):
    """
    A module to construct a fitness environment for certain algorithms 
    that follow reinforcement learning approach of optimization
    
    :param method: (str) the supported algorithms, choose either: ``ppo``, ``acktr``, ``a2c``
    :param fit: (function) the fitness function
    :param bounds: (dict) input parameter type and lower/upper bounds in dictionary form. Example: ``bounds={'x1': ['int', 1, 4], 'x2': ['float', 0.1, 0.8], 'x3': ['float', 2.2, 6.2]}``
    :param ncores: (int) number of parallel processors
    :param mode: (str) problem type, either ``min`` for minimization problem or ``max`` for maximization (RL is default to ``max``)
    
    :param selection: (str) type of sub-routine in PEARL to assign the rank of the solutions generated compared to the best solutions stored.
    :param sorting: (str) sorting type, ``standard`` or ``log``. The latter should be faster and is used as default.
    :param buffer_size: (int) number of solutions stored in the memory of each agent. The lowest rank assigned to a solution generated is therefore equal to buffer_size.
    :param: p: (int) number of divisions along each objective for the reference points. The number of reference points is Combination(M + p - 1, p), where M is the number of objective
    :param ref_points: (list) of user inputs reference points. If none the reference points are generated uniformly on the hyperplane intersecting each axis at 1. Utilised when niching and niching2 are used for selection.
    :param buffer_pop: (None or csv) utilized to store solutions and rank new ones. If a csv, the agent will start with old solutions generated, giving a head start to the algorithm in terms of which solutions are good.
    :param paradigm: (str) type of optimization, whether constrained or unconstrained.
    :param cl_penalty: (float) fixed constraints assigned to the reward of non-feasible solutions. We recommand it to be equal to buffer_size.
    :param nadir: (list) nadir (theoretical worst solution) used to calculate the hypervolume of the population of solutions found. Utilized in the callback.
    
    """
    
    def make_env(rank, seed=0):
        #"""
        #Utility function for multiprocessed env.
        # 
        #:param num_env: (int) the number of environment you wish to have in subprocesses
        #:param seed: (int) the inital seed for RNG
        #:param rank: (int) index of the subprocess
        #"""
        def _init():
            env=BaseEnvironment(method=method, fit=fit, 
                          bounds=bounds, mode=mode, selection=selection,sorting=sorting, p = p,buffer_size = buffer_size,ref_points = ref_points, paradigm=paradigm,cl_penalty=cl_penalty,buffer_pop=buffer_pop,nadir=nadir)
            env.seed(seed + rank)
            
            return env
        set_global_seeds(seed)
        return _init

    if ncores > 1:
        env = SubprocVecEnv([make_env(i) for i in range(ncores)])
    else:
        env=BaseEnvironment(method=method, fit=fit, 
                      bounds=bounds, mode=mode, selection=selection,sorting=sorting, p = p,buffer_size = buffer_size,ref_points = ref_points,paradigm=paradigm,cl_penalty=cl_penalty,buffer_pop=buffer_pop,nadir=nadir)
    return env


#########################################################################################################################
#  (Modified from NSGA II/III) Multi-Objective Related Functions
#  reference: [fortin2012deap] 
#   F\'elix-Antoine Fortin and Fran\c{c}ois-Michel De Rainville and Marc-Andr\'e Gardner and Marc Parizeau and Christian Gagn\'e " (2012),
#  'DEAP: Evolutionary Algorithms Made Easy', Journal of Machine Learning Research, pp 2171--2175, vol. 13, Jul. 2012.
#########################################################################################################################


def select(pop, k = 1, nd='standard',selection="crowding",ref_points=None, best_point=None,
             worst_point=None, extreme_points=None):
    """
    :param pop: (dict) A list of pop to select from.
    :param k: (int) The number of pop to select.
    :param nd: (str) Specify the non-dominated algorithm to use: 'standard' or 'log'.
    :Returns best_dict: (dict) ranked population in dictionary structure
    """
    if selection in ["niching2","crowding2"]:
        if nd == 'standard':
            pareto_fronts,flag = sortNondominated_constr(pop, k)
        elif nd == 'log':
            pareto_fronts,flag = sortLogNondominated_constr(pop, k)
        else:
            raise Exception('NSGA2: The choice of non-dominated sorting '
                            'method "{0}" is invalid.'.format(nd))
    else:
        if nd == 'standard':
            pareto_fronts = sortNondominated(pop, k)
        elif nd == 'log':
            pareto_fronts = sortLogNondominated(pop, k)
        else:
            raise Exception('NSGA2: The choice of non-dominated sorting '
                        'method "{0}" is invalid.'.format(nd))

    if selection in ["crowding","crowding2"]:
        if selection == "crowding2":
            if flag: # only unfeasible element
                best_dict=defaultdict(list)
                index=pareto_fronts[0][0][0]
                best_dict[index] = pareto_fronts[0][0][1]
                for key in pareto_fronts[1][:k-1]:
                    index = key[0]
                    best_dict[index] = key[1]
                return tuple(best_dict.items())
            
            feasible_index = 0
            for count,pareto in enumerate(pareto_fronts[0]):
                if np.sum(pareto[1][-1]) == 0:
                    feasible_index += 1
                else:
                    break
            if feasible_index <= 1:# either the first or the second one
                selected = pareto_fronts[0][feasible_index - 1]
                chosen = [tuple(selected)]
            else:
                if len(pareto_fronts[0][:feasible_index]) < k:
                    number_to_retrieve = len(pareto_fronts[0][:feasible_index])
                else:
                    number_to_retrieve = k
                CrowdDist = assignCrowdingDist_constr(pareto_fronts[0][:feasible_index])# the different fronts return are such that if add the last one then we would exceed 'k'
                # Order the pareto_front too
                sorted_front = sorted(CrowdDist,key = lambda k: CrowdDist[k],reverse=True)    
                sorted_front = [(x , pop[x]) for x in sorted_front][:number_to_retrieve]
                chosen = list(sorted_front)
            
            for fronts in pareto_fronts[0][feasible_index:]:# the last pareto front has too many samples. First order the current by front and crowding distances
                sel_count = len(chosen)
                if k - sel_count <= 0:
                    break
                chosen.extend([tuple(fronts)])

            if len(pareto_fronts[:-1]) != 0:# if one single front then sort it with crowd distance too:
                for fronts in pareto_fronts[1:-1]:# the last pareto front has too many samples. First order the current by front and crowding distances
                    # Use niching to select the remaining individuals
                    sel_count = len(chosen)
                    if k - sel_count <= 0:
                        break 
                    feasible_index = 0
                    for count,pareto in enumerate(fronts):
                        if np.sum(pareto[1][-1]) == 0:
                            feasible_index += 1
                        else:
                            break
                    if feasible_index <= 1:# either the first or the second one
                        selected = fronts[feasible_index - 1]
                        chosen.extend([tuple(selected)])
                    else:
                        if len(fronts[:feasible_index]) < k - sel_count:
                            number_to_retrieve = len(fronts[:feasible_index])
                        else:
                            number_to_retrieve = k - sel_count
                        CrowdDist = assignCrowdingDist_constr(fronts[:feasible_index])
                        # Order the pareto_front
                        sorted_front = sorted(CrowdDist,key = lambda k: CrowdDist[k],reverse=True)    
                        sorted_front = [(x , pop[x]) for x in sorted_front][:number_to_retrieve]
                        chosen.extend(list(sorted_front))

                    for subfronts in fronts[feasible_index:]:# this should happen at the last front
                        # Use niching to select the remaining individuals
                        sel_count = len(chosen)
                        if k - sel_count <= 0:
                            break
                        chosen.extend([tuple(subfronts)])
            sel_count = k - len(chosen)
            if sel_count > 0 and len(pareto_fronts[:-1]) != 0:# Then choose the remainings for ranking
                feasible_index = 0
                for count,pareto in enumerate(pareto_fronts[-1]):
                    if np.sum(pareto[1][-1]) == 0:
                        feasible_index += 1
                    else:
                        break
                if feasible_index <= 1:# either the first or the second one
                    selected = pareto_fronts[-1][feasible_index - 1]
                    chosen.extend([tuple(selected)])
                else:
                    if len(pareto_fronts[-1][:feasible_index]) < k:
                        number_to_retrieve = len(pareto_fronts[-1][:feasible_index])
                    else:
                        number_to_retrieve = k
                    CrowdDist = assignCrowdingDist_constr(pareto_fronts[-1][:feasible_index])# the different fronts return are such that if add the last one then we would exceed 'k'
                    # Order the pareto_front too
                    sorted_front = sorted(CrowdDist,key = lambda k: CrowdDist[k],reverse=True)    
                    sorted_front = [(x , pop[x]) for x in sorted_front][:number_to_retrieve]
                    chosen.extend(list(sorted_front))

                for fronts in pareto_fronts[-1][feasible_index:]:# the last pareto front has too many samples. First order the current by front and crowding distances
                    sel_count = len(chosen)
                    if k - sel_count <= 0:
                        break
                    chosen.extend([tuple(fronts)])
        else:
            CrowdDist = assignCrowdingDist(pareto_fronts[0])# the different fronts return are such that if add the last one then we would exceed 'k'
            # Order the pareto_front too
            sorted_front = sorted(CrowdDist,key = lambda k: CrowdDist[k],reverse=True)    
            sorted_front = [(x , pop[x]) for x in sorted_front]
            chosen = list(sorted_front)
            if len(pareto_fronts[:-1]) != 0:# if one single front then sort it with crowd distance too:
                for fronts in pareto_fronts[1:-1]:# the last pareto front has too many samples. First order the current by front and crowding distances
                    CrowdDist = assignCrowdingDist(fronts)
                    # Order the pareto_front
                    sorted_front = sorted(CrowdDist,key = lambda k: CrowdDist[k],reverse=True)    
                    sorted_front = [(x , pop[x]) for x in sorted_front]
                    chosen.extend(list(sorted_front))


            k = k - len(chosen)
            if k > 0 and len(pareto_fronts[:-1]) != 0:# Then choose the remainings for ranking
                CrowdDist = assignCrowdingDist(pareto_fronts[-1])
                sorted_front = sorted(CrowdDist,key = lambda k: CrowdDist[k],reverse=True)    
                sorted_front = [(x , pop[x]) for x in sorted_front]
                chosen.extend(sorted_front[:k])

    elif selection in ["niching","niching2"]:
        if selection in ["niching2"]:
            # Extract fitnesses as a np array in the nd-sort order
            # Use * -1 to tackle always as a minimization problem. Necessary here as well
            if flag: # only unfeasible element
                best_dict=defaultdict(list)
                index=pareto_fronts[0][0][0]
                best_dict[index] = pareto_fronts[0][0][1]
                for key in pareto_fronts[1][:k-1]:
                    index = key[0]
                    best_dict[index] = key[1]
                return tuple(best_dict.items())
            
            fitnesses = np.array([ind[1][1]['objectives'] for f in pareto_fronts for ind in f if np.sum(ind[1][1]['constraints']) == 0])# else np.array(ind[1][1]['objectives']) - 1e11 for f in pareto_fronts for ind in f])#np.array([ind[1][2]['objectives'] for f in pareto_fronts for ind in f if np.sum(ind[1][2]['constraints']) == 0])
            fitnesses *= -1
        else:
            fitnesses = np.array([ind[1][1] for f in pareto_fronts for ind in f])
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
        index, counts = np.unique(niches, return_counts=True)
        niche_counts[index] = counts
        if len(pareto_fronts[0]) < k:
            number_to_retrieve = len(pareto_fronts[0])
        else:
            number_to_retrieve = k
        if selection not in ["niching2"]:
            selected = niching(pareto_fronts[0], number_to_retrieve, niches[:len(pareto_fronts[0])], dist[:len(pareto_fronts[0])], niche_counts)
            chosen = list(selected)
            # Use niching to select the remaining individuals
            for fronts in pareto_fronts[1:]:# the last pareto front has too many samples. First order the current by front and crowding distances
                # Use niching to select the remaining individuals
                sel_count = len(chosen)
                if k - sel_count == 0:
                    break
                if len(fronts) < k - sel_count:
                    number_to_retrieve = len(fronts)
                else:
                    number_to_retrieve = k - sel_count
                selected = niching(fronts, number_to_retrieve, niches[sel_count:sel_count+len(fronts)], dist[sel_count:sel_count+len(fronts)], niche_counts)
                chosen.extend(selected)
        
        else:
            feasible_index = 0
            for count,pareto in enumerate(pareto_fronts[0]):
                if np.sum(pareto[1][-1]) == 0:
                    feasible_index += 1
                else:
                    break
            if feasible_index <= 1:# either the first or the second one
                selected = pareto_fronts[0][feasible_index - 1]
                chosen = [tuple(selected)]
            else:
                if len(pareto_fronts[0][:feasible_index]) < k:
                    number_to_retrieve = len(pareto_fronts[0][:feasible_index])
                else:
                    number_to_retrieve = k
                if feasible_index < number_to_retrieve:
                    selected = niching(pareto_fronts[0][:feasible_index], number_to_retrieve, niches[:len(pareto_fronts[0][:feasible_index])], dist[:len(pareto_fronts[0][:feasible_index])], niche_counts) 
                selected = niching(pareto_fronts[0][:feasible_index], number_to_retrieve, niches[:len(pareto_fronts[0][:feasible_index])], dist[:len(pareto_fronts[0][:feasible_index])], niche_counts) 
                chosen = list(selected)

            for fronts in pareto_fronts[0][feasible_index:]:# the last pareto front has too many samples. First order the current by front and crowding distances
                # Use niching to select the remaining individuals
                sel_count = len(chosen)
                if k - sel_count == 0:
                    break
                chosen.extend([tuple(fronts)])
            # Use niching to select the remaining individuals
            for fronts in pareto_fronts[1:]:# the last pareto front has too many samples. First order the current by front and crowding distances
                # Use niching to select the remaining individuals
                sel_count = len(chosen)
                if k - sel_count == 0:
                    break
                
                feasible_index = 0
                for count,pareto in enumerate(fronts):
                    if np.sum(pareto[1][-1]) == 0:
                        feasible_index += 1
                    else:
                        break
                if feasible_index <= 1:# either the first or the second one
                    selected = fronts[feasible_index - 1]
                    chosen.extend([tuple(selected)])
                else:
                    if len(fronts[:feasible_index]) < k - sel_count:
                        number_to_retrieve = len(fronts[:feasible_index])
                    else:
                        number_to_retrieve = k - sel_count
                    selected = niching(fronts[:feasible_index], number_to_retrieve, niches[sel_count:sel_count+len(fronts[:feasible_index])], dist[sel_count:sel_count+len(fronts[:feasible_index])], niche_counts)
                    chosen.extend(selected)
                for subfronts in fronts[feasible_index:]:# this should happen at the last front
                    # Use niching to select the remaining individuals
                    sel_count = len(chosen)
                    if k - sel_count == 0:
                        break
                    chosen.extend([tuple(subfronts)])
    return chosen

##############################################################
# Helper functions for sorting individuals in the population #
##############################################################
from operator import itemgetter
import bisect
from collections import defaultdict
from itertools import chain

def isDominated(wvalues1, wvalues2):
    """
    Returns whether or not *wvalues2* dominates *wvalues1*.
    :param wvalues1: (list) The weighted fitness values that would be dominated.
    :param wvalues2: (list) The weighted fitness values of the dominant.
    :Returns obj: (bool) `True` if wvalues2 dominates wvalues1, `False` otherwise.
    
    """
    not_equal = False
    for self_wvalue, other_wvalue in zip(wvalues1, wvalues2):
        if self_wvalue > other_wvalue:    
            return False
        elif self_wvalue < other_wvalue:
            not_equal = True
    return not_equal

def sortNondominated(pop, k, first_front_only=False):
    """
    Sort the first *k* *pop* into different nondomination levels.
    
    :param pop: (dict) population in dictionary structure
    :param k: (int) top k individuals are selected
    :param first_front_only [DEPRECATED]: (bool) If :obj:`True` sort only the first front and exit. 
    :Returns pareto_front: (list) A list of Pareto fronts, the first list includes nondominated pop.
    .. 
    reference: [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
    non-dominated sorting genetic algorithm for multi-objective
    optimization: NSGA-II", 2002.
    """
    if k == 0:
        return []

    pop=list(pop.items())

    map_fit_ind = defaultdict(list)
    for ind in pop:
        map_fit_ind[ind[0]].append(ind)
    fits = list(map_fit_ind.keys())

    current_front = []
    next_front = []
    dominating_fits = defaultdict(int)
    dominated_fits = defaultdict(list)

    # Rank first Pareto front
    for i, fit_i in enumerate(fits):
        for fit_j in fits[i+1:]:
            if isDominated(map_fit_ind[fit_j][0][1][1], map_fit_ind[fit_i][0][1][1]):
                dominating_fits[fit_j] += 1
                dominated_fits[fit_i].append(fit_j)
            elif isDominated(map_fit_ind[fit_i][0][1][1], map_fit_ind[fit_j][0][1][1]):
                dominating_fits[fit_i] += 1
                dominated_fits[fit_j].append(fit_i)
        if dominating_fits[fit_i] == 0:
            current_front.append(fit_i)

    fronts = [[]]
    for fit in current_front:
        fronts[-1].extend(map_fit_ind[fit])
    pareto_sorted = len(fronts[-1])

    # Rank the next front until all pop are sorted or
    # the given number of individual are sorted.
    if not first_front_only:
        N = min(len(pop), k)
        while pareto_sorted < N:
            fronts.append([])
            for fit_p in current_front:
                for fit_d in dominated_fits[fit_p]:
                    dominating_fits[fit_d] -= 1
                    if dominating_fits[fit_d] == 0:
                        next_front.append(fit_d)
                        pareto_sorted += len(map_fit_ind[fit_d]) # add element to the next solution
                        fronts[-1].extend(map_fit_ind[fit_d])
            current_front = next_front
            next_front = []

    return fronts

#######################################
# Generalized Reduced runtime ND sort #
#######################################

def identity(obj):
    """
    Returns directly the argument *obj*.
    :param obj: (type)
    :Returns obj: (type)
    """
    return obj


def median(seq, key=identity):
    """
    Returns the median of *seq* - the numeric value separating the higher
    half of a sample from the lower half. If there is an even number of
    elements in *seq*, it returns the mean of the two middle values.
    """
    sseq = sorted(seq, key=key)
    length = len(seq)
    if length % 2 == 1:
        return key(sseq[(length - 1) // 2])
    else:
        return (key(sseq[(length - 1) // 2]) + key(sseq[length // 2])) / 2.0

def sortLogNondominated(pop, k, first_front_only=False):
    """
    Sort the first *k* *pop* into different nondomination levels.
    
    :param pop: (dict) population in dictionary structure
    :param k: (int) top k individuals are selected
    :param first_front_only [DEPRECATED]: (bool) If :obj:`True` sort only the first front and exit. 
    :Returns pareto_front: (list) A list of Pareto fronts, the first list includes nondominated pop.
    .. 
    reference: [Fortin2013] Fortin, Grenier, Parizeau, "Generalizing the improved run-time complexity algorithm for non-dominated sorting",
    Proceedings of the 15th annual conference on Genetic and evolutionary computation, 2013. 
    """
    if k == 0:
        return []

    pop=list(pop.items())
    #Separate individuals according to unique fitnesses
    unique_fits = defaultdict(list)
    for i, ind in enumerate(pop):
        unique_fits[tuple(ind[1][1])].append(ind)
            

    #Launch the sorting algorithm
    obj = len(pop[0][1][1])-1
    fitnesses = list(unique_fits.keys())
    front = dict.fromkeys(fitnesses, 0)

    # Sort the fitnesses lexicographically.
    fitnesses.sort(reverse=True)
    sortNDHelperA(fitnesses, obj, front)

    #Extract pop from front list here
    nbfronts = max(front.values())+1
    pareto_fronts = [[] for i in range(nbfronts)]
    for fit in fitnesses:
        index = front[fit]
        pareto_fronts[index].extend(unique_fits[fit])

    # Keep only the fronts required to have k pop.
    if not first_front_only:
        count = 0
        for i, front in enumerate(pareto_fronts):
            count += len(front)
            if count >= k:
                return pareto_fronts[:i+1]
        return pareto_fronts
    else:
        return pareto_fronts[0]

def sortNDHelperA(fitnesses, obj, front):
    """
    Create a non-dominated sorting of S on the first M objectives
    """
    if len(fitnesses) < 2:
        return
    elif len(fitnesses) == 2:
        # Only two individuals, compare them and adjust front number
        s1, s2 = fitnesses[0], fitnesses[1]
        if isDominated(s2[:obj+1], s1[:obj+1]):
            front[s2] = max(front[s2], front[s1] + 1)
    elif obj == 1:
        sweepA(fitnesses, front)
    elif len(frozenset(map(itemgetter(obj), fitnesses))) == 1:
        #All individuals for objective M are equal: go to objective M-1
        sortNDHelperA(fitnesses, obj-1, front)
    else:
        # More than two individuals, split list and then apply recursion
        best, worst = splitA(fitnesses, obj)
        sortNDHelperA(best, obj, front)
        sortNDHelperB(best, worst, obj-1, front)
        sortNDHelperA(worst, obj, front)

def splitA(fitnesses, obj):
    """
    Partition the set of fitnesses in two according to the median of
    the objective index *obj*. The values equal to the median are put in
    the set containing the least elements.
    """
    median_ = median(fitnesses, itemgetter(obj))
    best_a, worst_a = [], []
    best_b, worst_b = [], []

    for fit in fitnesses:
        if fit[obj] > median_:
            best_a.append(fit)
            best_b.append(fit)
        elif fit[obj] < median_:
            worst_a.append(fit)
            worst_b.append(fit)
        else:
            best_a.append(fit)
            worst_b.append(fit)

    balance_a = abs(len(best_a) - len(worst_a))
    balance_b = abs(len(best_b) - len(worst_b))

    if balance_a <= balance_b:
        return best_a, worst_a
    else:
        return best_b, worst_b

def sweepA(fitnesses, front):
    """
    Update rank number associated to the fitnesses according
    to the first two objectives using a geometric sweep procedure.
    """
    stairs = [-fitnesses[0][1]]
    fstairs = [fitnesses[0]]
    for fit in fitnesses[1:]:
        idx = bisect.bisect_right(stairs, -fit[1])
        if 0 < idx <= len(stairs):
            fstair = max(fstairs[:idx], key=front.__getitem__)
            front[fit] = max(front[fit], front[fstair]+1)
        for i, fstair in enumerate(fstairs[idx:], idx):
            if front[fstair] == front[fit]:
                del stairs[i]
                del fstairs[i]
                break
        stairs.insert(idx, -fit[1])
        fstairs.insert(idx, fit)

def sortNDHelperB(best, worst, obj, front):
    """
    Assign front numbers to the solutions in H according to the solutions
    in L. The solutions in L are assumed to have correct front numbers and the
    solutions in H are not compared with each other, as this is supposed to
    happen after sortNDHelperB is called.
    """
    key = itemgetter(obj)
    if len(worst) == 0 or len(best) == 0:
        #One of the lists is empty: nothing to do
        return
    elif len(best) == 1 or len(worst) == 1:
        #One of the lists has one individual: compare directly
        for hi in worst:
            for li in best:
                if isDominated(hi[:obj+1], li[:obj+1]) or hi[:obj+1] == li[:obj+1]:
                    front[hi] = max(front[hi], front[li] + 1)
    elif obj == 1:
        sweepB(best, worst, front)
    elif key(min(best, key=key)) >= key(max(worst, key=key)):
        #All individuals from L dominate H for objective M:
        #Also supports the case where every individuals in L and H
        #has the same value for the current objective
        #Skip to objective M-1
        sortNDHelperB(best, worst, obj-1, front)
    elif key(max(best, key=key)) >= key(min(worst, key=key)):
        best1, best2, worst1, worst2 = splitB(best, worst, obj)
        sortNDHelperB(best1, worst1, obj, front)
        sortNDHelperB(best1, worst2, obj-1, front)
        sortNDHelperB(best2, worst2, obj, front)

def splitB(best, worst, obj):
    """
    Split both best individual and worst sets of fitnesses according
    to the median of objective *obj* computed on the set containing the
    most elements. The values equal to the median are attributed so as
    to balance the four resulting sets as much as possible.
    """
    median_ = median(best if len(best) > len(worst) else worst, itemgetter(obj))
    best1_a, best2_a, best1_b, best2_b = [], [], [], []
    for fit in best:
        if fit[obj] > median_:
            best1_a.append(fit)
            best1_b.append(fit)
        elif fit[obj] < median_:
            best2_a.append(fit)
            best2_b.append(fit)
        else:
            best1_a.append(fit)
            best2_b.append(fit)

    worst1_a, worst2_a, worst1_b, worst2_b = [], [], [], []
    for fit in worst:
        if fit[obj] > median_:
            worst1_a.append(fit)
            worst1_b.append(fit)
        elif fit[obj] < median_:
            worst2_a.append(fit)
            worst2_b.append(fit)
        else:
            worst1_a.append(fit)
            worst2_b.append(fit)

    balance_a = abs(len(best1_a) - len(best2_a) + len(worst1_a) - len(worst2_a))
    balance_b = abs(len(best1_b) - len(best2_b) + len(worst1_b) - len(worst2_b))

    if balance_a <= balance_b:
        return best1_a, best2_a, worst1_a, worst2_a
    else:
        return best1_b, best2_b, worst1_b, worst2_b

def sweepB(best, worst, front):
    """
    Adjust the rank number of the worst fitnesses according to
    the best fitnesses on the first two objectives using a sweep
    procedure.
    """
    stairs, fstairs = [], []
    iter_best = iter(best)
    next_best = next(iter_best, False)
    for h in worst:
        while next_best and h[:2] <= next_best[:2]:
            insert = True
            for i, fstair in enumerate(fstairs):
                if front[fstair] == front[next_best]:
                    if fstair[1] > next_best[1]:
                        insert = False
                    else:
                        del stairs[i], fstairs[i]
                    break
            if insert:
                idx = bisect.bisect_right(stairs, -next_best[1])
                stairs.insert(idx, -next_best[1])
                fstairs.insert(idx, next_best)
            next_best = next(iter_best, False)

        idx = bisect.bisect_right(stairs, -h[1])
        if 0 < idx <= len(stairs):
            fstair = max(fstairs[:idx], key=front.__getitem__)
            front[h] = max(front[h], front[fstair]+1)


##########################################################################
# niching - based Selection functions 
# reference: [Deb2014] Deb, K., & Jain, H. (2014). 
# An Evolutionary Many-Objective Optimization
# Algorithm Using Reference-Point-Based Nondominated Sorting Approach,
# Part I: Solving Problems With Box Constraints. IEEE Transactions on
# Evolutionary Computation, 18(4), 577-601. doi:10.1109/TEVC.2013.2281535.
##########################################################################

def find_extreme_points(fitnesses, best_point, extreme_points=None):
    """
    Finds the individuals with extreme values for each objective function.
    :param fitnesses: (list) list of fitness for each individual
    :param best_point: (list) list of the best fitness found for each objective
    :param extreme_points: (list) Extreme points found at previous generation. If not provided
    find the extreme points only from current pop.  

    :Returns fitness with minimal asf (new extreme points)
    """
    # Keep track of last generation extreme points
    if extreme_points is not None:
        fitnesses = np.concatenate((fitnesses, extreme_points), axis=0)

    # Translate objectives
    ft = fitnesses - best_point

    # Find achievement scalarizing function (asf)
    asf = np.eye(best_point.shape[0])
    asf[asf == 0] = 1e6
    asf = np.max(ft * asf[:, np.newaxis, :], axis=2)

    # Extreme point are the fitnesses with minimal asf
    min_asf_idx = np.argmin(asf, axis=1)
    return fitnesses[min_asf_idx, :]


def find_intercepts(extreme_points, best_point, current_worst, front_worst):
    """
    Find intercepts between the hyperplane and each axis with
    the ideal point as origin.

    :param extreme_points: (list) list of extreme points for each objective
    :param best_point: (list) list of best points for each objective
    :param current_worst: (list) list of worst fitness for each objective (ever found if memory is implemented: TO DO!!)
    :Param front_worst: (list) current list of worst fitness for each objective (equal to current_worst for the current version)

    :Returns intercepts: (list) Obj-dimensional intercept.
    """
    # Construct hyperplane sum(f_i^n) = 1
    b = np.ones(extreme_points.shape[1])
    A = extreme_points - best_point
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        intercepts = current_worst
    else:
        if np.count_nonzero(x) != len(x):
            intercepts = front_worst
        else:
            intercepts = 1 / x

            if (not np.allclose(np.dot(A, x), b) or
                    np.any(intercepts <= 1e-6) or
                    np.any((intercepts + best_point) > current_worst)):
                intercepts = front_worst

    return intercepts


def associate_to_niche(fitnesses, reference_points, best_point, intercepts):
    """
    Associates individuals to reference points and calculates niche number.
    Corresponds to Algorithm 3 of Deb & Jain (2014).
    
    :param fitnesses: (list) list of fitness for each individual
    :param reference_points: (list) list of Obj-dimensional reference points leveraged to obtain a well-spread pareto front
    :param best_point: (list) list of the best fitness found for each objective
    :param intercepts: (list) Obj-dimensional intercept.
    :Returns
    niches: (list) associated reference point for each individual
    distances: (list) distance of each individual to its associated niche
    """
    # Normalize by ideal point and intercepts
    fn = (fitnesses - best_point) / (intercepts - best_point)

    # Create distance matrix
    fn = np.repeat(np.expand_dims(fn, axis=1), len(reference_points), axis=1)
    norm = np.linalg.norm(reference_points, axis=1)

    distances = np.sum(fn * reference_points, axis=2) / norm.reshape(1, -1)
    distances = distances[:, :, np.newaxis] * reference_points[np.newaxis, :, :] / norm[np.newaxis, :, np.newaxis]
    distances = np.linalg.norm(distances - fn, axis=2)

    # Retrieve min distance niche index
    niches = np.argmin(distances, axis=1)
    distances = distances[range(niches.shape[0]), niches]
    return niches, distances


def niching(pop, k, niches, distances, niche_counts):
    """
    niche preserving operator. Choose elements which are associated to reference points
    with the lowest number of association from the already choosen batch of solution Pt
    
    :param pop: (dict) population in dictionary structure
    :param k: (int) top k individuals to select to complete the population
    :param niches: (list) associated reference point for each individual
    :param distances: (list) distance of each individual to its associated niche
    :param niching: (list) count per niche 

    :Returns selected: (list) remaining individual to complete the population
    """
    selected = []
    available = np.ones(len(pop), dtype=np.bool)
    while len(selected) < k:
        # Maximum number of individuals (niches) to select in that round
        n = k - len(selected)

        # Find the available niches and the minimum niche count in them
        available_niches = np.zeros(len(niche_counts), dtype=np.bool)
        available_niches[np.unique(niches[available])] = True
        min_count = np.min(niche_counts[available_niches])

        # Select at most n niches with the minimum count
        selected_niches = np.flatnonzero(np.logical_and(available_niches, niche_counts == min_count))
        np.random.shuffle(selected_niches)
        selected_niches = selected_niches[:n]

        for niche in selected_niches:
            # Select from available individuals in niche
            niche_individuals = np.flatnonzero(np.logical_and(niches == niche, available))
            np.random.shuffle(niche_individuals)

            # If no individual in that niche, select the closest to reference
            # Else select randomly
            if niche_counts[niche] == 0:
                sel_index = niche_individuals[np.argmin(distances[niche_individuals])]
            else:
                sel_index = niche_individuals[0]

            # Update availability, counts and selection
            available[sel_index] = False
            niche_counts[niche] += 1
            selected.append(pop[sel_index])

    return selected


def uniform_reference_points(nobj, p=4, scaling=None):
    """
    Generate reference points uniformly on the hyperplane intersecting
    each axis at 1. The scaling factor is used to combine multiple layers of
    reference points.

    :param nobj: (int) number of objective
    :param p: (int) number of division along each objective
    :param scaling [DEPRECATED]:
    :Returns ref_points: (list) list of Obj-dimensional reference points 
    """
    def gen_refs_recursive(ref, nobj, left, total, depth):
        points = []
        if depth == nobj - 1:
            ref[depth] = left / total
            points.append(ref)
        else:
            for i in range(left + 1):
                ref[depth] = i / total
                points.extend(gen_refs_recursive(ref.copy(), nobj, left - i, total, depth + 1))
        return points

    ref_points = np.array(gen_refs_recursive(np.zeros(nobj), nobj, p, p, 0))
    if scaling is not None:
        ref_points *= scaling
        ref_points += (1 - scaling) / nobj

    return ref_points


##########################################################################
# crowding distance - based Selection functions 
# reference: [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
# non-dominated sorting genetic algorithm for multi-objective
# optimization: NSGA-II", 2002.
##########################################################################

def assignCrowdingDist(pop):
    """
    Assign a crowding distance to each individual's fitness. 

    :param pop: (list) list of individuals and assocated positions, strategy vector, and fitness
    :Returns:
    CrowDist: (dict) dictionnary of element of pop and associated crowding distance
    """
    if len(pop) == 0:
        return
    CrowdDist = {}
    distances = [0.0] * len(pop)
    crowd = [(ind[1][1], i) for i, ind in enumerate(pop)]
    nobj = len(pop[0][1][1])

    for i in range(nobj):
        crowd.sort(key=lambda element: element[0][i])
        distances[crowd[0][1]] = float("inf")
        distances[crowd[-1][1]] = float("inf")
        if crowd[-1][0][i] == crowd[0][0][i]:
            continue
        norm = nobj * float(crowd[-1][0][i] - crowd[0][0][i])
        for prev, cur, next in zip(crowd[:-2], crowd[1:-1], crowd[2:]):
            distances[cur[1]] += (next[0][i] - prev[0][i]) / norm

    for i, dist in enumerate(distances):
        CrowdDist[pop[i][0]] = dist
    return CrowdDist

def assignCrowdingDist_constr(pop):
    """
    Assign a crowding distance to each individual's fitness. 

    :param pop: (list) list of individuals and assocated positions, strategy vector, and fitness
    :Returns:
    CrowDist: (dict) dictionnary of element of pop and associated crowding distance
    """
    if len(pop) == 0:
        return
    CrowdDist = {}
    distances = [0.0] * len(pop)
    crowd = [(ind[1][1]["objectives"], i) for i, ind in enumerate(pop)]
    nobj = len(pop[0][1][1])

    for i in range(nobj):
        crowd.sort(key=lambda element: element[0][i])
        distances[crowd[0][1]] = float("inf")
        distances[crowd[-1][1]] = float("inf")
        if crowd[-1][0][i] == crowd[0][0][i]:
            continue
        norm = nobj * float(crowd[-1][0][i] - crowd[0][0][i])
        for prev, cur, next in zip(crowd[:-2], crowd[1:-1], crowd[2:]):
            distances[cur[1]] += (next[0][i] - prev[0][i]) / norm

    for i, dist in enumerate(distances):
        CrowdDist[pop[i][0]] = dist
    return CrowdDist

def isDominated_constr(wvalues1, wvalues2):
    """
    
    Returns whether or not *wvalues2* dominates *wvalues1* with a modification to handle constraints. (see 
    "An Evolutionary Many-Objective Optimization Algorithm Using Reference-point Based Non-dominated Sortign Approach, 
    Part II: Handling Constraints and Extending to an Adaptive Approach" Jain et al. (2013))
    
    :param wvalues1: (list) The weighted fitness values that would be dominated.
    :param wvalues2: (list) The weighted fitness values of the dominant.
    :Returns obj: (bool) `True` if wvalues2 dominates wvalues1, `False` otherwise.
    
    """
    not_equal = False
    if np.sum(wvalues1["constraints"]) == 0 and np.sum(wvalues2["constraints"]) != 0:
        return False
    elif np.sum(wvalues1["constraints"]) != 0 and np.sum(wvalues2["constraints"]) == 0:
        not_equal = True
    elif np.sum(wvalues1["constraints"]) != 0 and np.sum(wvalues2["constraints"]) != 0:
        if np.sum(wvalues1["constraints"]) > np.sum(wvalues2["constraints"]):
            not_equal = True
        else:
            not_equal = False
    elif np.sum(wvalues1["constraints"]) == 0 and np.sum(wvalues2["constraints"]) == 0:
        not_equal = isDominated(wvalues1["objectives"], wvalues2["objectives"])
    return not_equal


def sortNondominated_constrX(pop, k, first_front_only=False):
    """
    Sort the first *k* *pop* into different nondomination levels.
    
    :param pop: (dict) population in dictionary structure
    :param k: (int) top k individuals are selected
    :param first_front_only [DEPRECATED]: (bool) If :obj:`True` sort only the first front and exit. 
    :Returns pareto_front: (list) A list of Pareto fronts, the first list includes nondominated pop.
    .. 

    reference: [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
    non-dominated sorting genetic algorithm for multi-objective
    optimization: NSGA-II", 2002.
    """
    if k == 0:
        return []

    pop=list(pop.items())

    map_fit_ind = defaultdict(list)
    unfeasible_fits = defaultdict(list)
    for ind in pop:
        if np.sum(ind[1][2]) == 0:
            map_fit_ind[ind[0]].append(ind)
        else:
            unfeasible_fits[ind[0]].append(ind)
    fits = list(map_fit_ind.keys())
    if fits == []:
        return [list(sorted(pop,key=lambda x: np.sum(x[1][1])))],True#x[1][1][0]
    else:
        map_fit_ind.update(unfeasible_fits)
        fits = list(map_fit_ind.keys())
    
    current_front = []
    next_front = []
    dominating_fits = defaultdict(int)
    dominated_fits = defaultdict(list)

    # Rank first Pareto front
    for i, fit_i in enumerate(fits):
        for fit_j in fits[i+1:]:
            A = {"objectives":map_fit_ind[fit_j][0][1][1],"constraints":map_fit_ind[fit_j][0][1][2]}
            B = {"objectives":map_fit_ind[fit_i][0][1][1],"constraints":map_fit_ind[fit_i][0][1][2]}
            if isDominated_constr(A,B):#
                dominating_fits[fit_j] += 1
                dominated_fits[fit_i].append(fit_j)
            elif isDominated_constr(B,A):#
                dominating_fits[fit_i] += 1
                dominated_fits[fit_j].append(fit_i)
        if dominating_fits[fit_i] == 0:
            current_front.append(fit_i)
    

    fronts = [[]]
    for fit in current_front:
        fronts[-1].extend(map_fit_ind[fit])

    pareto_sorted = len(fronts[-1])

    # Rank the next front until all pop are sorted or
    # the given number of individual are sorted.
    if not first_front_only:
        N = min(len(pop), k)
        while pareto_sorted < N:
            fronts.append([])
            for fit_p in current_front:
                for fit_d in dominated_fits[fit_p]:
                    dominating_fits[fit_d] -= 1
                    if dominating_fits[fit_d] == 0:
                        next_front.append(fit_d)
                        pareto_sorted += len(map_fit_ind[fit_d]) # add element to the next solution
                        fronts[-1].extend(map_fit_ind[fit_d])
            current_front = next_front
            next_front = []
    return fronts,False

def sortNondominated_constr(pop, k, first_front_only=False):
    """
    Sort the first *k* *pop* into different nondomination levels.
    
    :param pop: (dict) population in dictionary structure
    :param k: (int) top k individuals are selected
    :param first_front_only [DEPRECATED]: (bool) If :obj:`True` sort only the first front and exit. 
    :Returns pareto_front: (list) A list of Pareto fronts, the first list includes nondominated pop.
    .. 

    reference: [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
    non-dominated sorting genetic algorithm for multi-objective
    optimization: NSGA-II", 2002.
    """
    if k == 0:
        return []

    pop=list(pop.items())
    map_fit_ind = defaultdict(list)
    unfeasible_fits = defaultdict(list)
    for ind in pop:
        if np.sum(tuple(ind[1][1].items())[1][1]) == 0:# add if feasible
            map_fit_ind[ind[0]].append(ind)
        else:
            unfeasible_fits[ind[0]].append(ind)
    fits = list(map_fit_ind.keys())
    if fits == []:
        return [[list(sorted(pop,key=lambda x: np.sum(tuple(x[1][1].items())[1][1])))[0]],list(sorted(pop,key=lambda x: np.sum(tuple(x[1][1].items())[1][1])))[1:]],True
    else:
        map_fit_ind.update(unfeasible_fits)
        fits = list(map_fit_ind.keys())
    
    current_front = []
    next_front = []
    dominating_fits = defaultdict(int)
    dominated_fits = defaultdict(list)

    # Rank first Pareto front
    for i, fit_i in enumerate(fits):
        for fit_j in fits[i+1:]:
            A = {"objectives":map_fit_ind[fit_j][0][1][1],"constraints":map_fit_ind[fit_j][0][1][2]}
            B = {"objectives":map_fit_ind[fit_i][0][1][1],"constraints":map_fit_ind[fit_i][0][1][2]}
            
            if isDominated_constr(A,B):#
                dominating_fits[fit_j] += 1
                dominated_fits[fit_i].append(fit_j)
            elif isDominated_constr(B,A):#
                dominating_fits[fit_i] += 1
                dominated_fits[fit_j].append(fit_i)
        if dominating_fits[fit_i] == 0:
            current_front.append(fit_i)
    

    fronts = [[]]
    for fit in current_front:
        fronts[-1].extend(map_fit_ind[fit])

    pareto_sorted = len(fronts[-1])

    # Rank the next front until all pop are sorted or
    # the given number of individual are sorted.
    if not first_front_only:
        N = min(len(pop), k)
        while pareto_sorted < N:
            fronts.append([])
            for fit_p in current_front:
                for fit_d in dominated_fits[fit_p]:
                    dominating_fits[fit_d] -= 1
                    if dominating_fits[fit_d] == 0:
                        next_front.append(fit_d)
                        pareto_sorted += len(map_fit_ind[fit_d]) # add element to the next solution
                        fronts[-1].extend(map_fit_ind[fit_d])
            current_front = next_front
            next_front = []
    return fronts,False

def sortNDHelperA_constr(fitnesses, obj, front):
    """
    Create a non-dominated sorting of S on the first M objectives
    """
    if len(fitnesses) < 2:
        return
    elif len(fitnesses) == 2:
        # Only two individuals, compare them and adjust front number
        s1, s2 = fitnesses[0], fitnesses[1]
        if isDominated(s2[:obj+1], s1[:obj+1]):
            front[s2] = max(front[s2], front[s1] + 1)
    elif obj == 1:
        sweepA(fitnesses, front)
    elif len(frozenset(map(itemgetter(obj), fitnesses))) == 1:
        #All individuals for objective M are equal: go to objective M-1
        sortNDHelperA_constr(fitnesses, obj-1, front)
    else:
        # More than two individuals, split list and then apply recursion
        best, worst = splitA(fitnesses, obj)
        sortNDHelperA_constr(best, obj, front)
        sortNDHelperB_constr(best, worst, obj-1, front)
        sortNDHelperA_constr(worst, obj, front)


def sortNDHelperB_constr(best, worst, obj, front):
    """
    Assign front numbers to the solutions in H according to the solutions
    in L. The solutions in L are assumed to have correct front numbers and the
    solutions in H are not compared with each other, as this is supposed to
    happen after sortNDHelperB is called.
    """
    key = itemgetter(obj)
    if len(worst) == 0 or len(best) == 0:
        #One of the lists is empty: nothing to do
        return
    elif len(best) == 1 or len(worst) == 1:
        #One of the lists has one individual: compare directly
        for hi in worst:
            for li in best:
                if isDominated(hi[:obj+1], li[:obj+1]) or hi[:obj+1] == li[:obj+1]:
                    front[hi] = max(front[hi], front[li] + 1)
    elif obj == 1:
        sweepB(best, worst, front)
    elif key(min(best, key=key)) >= key(max(worst, key=key)):
        #All individuals from L dominate H for objective M:
        #Also supports the case where every individuals in L and H
        #has the same value for the current objective
        #Skip to objective M-1
        sortNDHelperB_constr(best, worst, obj-1, front)
    elif key(max(best, key=key)) >= key(min(worst, key=key)):
        best1, best2, worst1, worst2 = splitB(best, worst, obj)
        sortNDHelperB_constr(best1, worst1, obj, front)
        sortNDHelperB_constr(best1, worst2, obj-1, front)
        sortNDHelperB_constr(best2, worst2, obj, front)


def sortLogNondominated_constrX(pop, k, first_front_only=False):
    """
    Sort the first *k* *pop* into different nondomination levels.
    
    :param pop: (dict) population in dictionary structure
    :param k: (int) top k individuals are selected
    :param first_front_only [DEPRECATED]: (bool) If :obj:`True` sort only the first front and exit. 
    :Returns pareto_front: (list) A list of Pareto fronts, the first list includes nondominated pop.
    .. 

    reference: [Fortin2013] Fortin, Grenier, Parizeau, "Generalizing the improved run-time complexity algorithm for non-dominated sorting",
    Proceedings of the 15th annual conference on Genetic and evolutionary computation, 2013. 
    """
    if k == 0:
        return []

    pop=list(pop.items())
    #Separate individuals according to unique fitnesses
    unique_fits = defaultdict(list)
    unfeasible_fits = [] # useful for NSGA that needs all elements even unfeasibles
    for i, ind in enumerate(pop):
        if np.sum(ind[1][2]) == 0:# add if feasible
            unique_fits[tuple(ind[1][1])].append(ind)
        else:
            unfeasible_fits.append(ind)
    #Launch the sorting algorithm
    obj = len(pop[0][1][1])-1
    fitnesses = list(unique_fits.keys())
    front = dict.fromkeys(fitnesses, 0)
    if fitnesses == []:
        return [list(sorted(pop,key=lambda x: np.sum(x[1][1])))],True#x[1][1][0]
    # Sort the fitnesses lexicographically.
    fitnesses.sort(reverse=True)
    sortNDHelperA_constr(fitnesses, obj, front)

    #Extract pop from front list here
    nbfronts = max(front.values())+1
    pareto_fronts = [[] for i in range(nbfronts)]
    for fit in fitnesses:
        index = front[fit]
        pareto_fronts[index].extend(unique_fits[fit])
    
    # added for NSGA-III 
    unfeasible_fits = list(sorted(unfeasible_fits,key=lambda x: np.sum(x[1][1])))# sort the unfeasible
    index = nbfronts - 1
    for fit in unfeasible_fits:
        pareto_fronts[index].extend([fit])
    # Keep only the fronts required to have k pop.
    if not first_front_only:
        count = 0
        for i, front in enumerate(pareto_fronts):
            count += len(front)
            if count >= k:

                return pareto_fronts[:i+1],False
        return pareto_fronts,False
    else:
        return pareto_fronts[0],False

def sortLogNondominated_constr(pop, k, first_front_only=False):
    """
    Sort the first *k* *pop* into different nondomination levels.
    
    :param pop: (dict) population in dictionary structure
    :param k: (int) top k individuals are selected
    :param first_front_only [DEPRECATED]: (bool) If :obj:`True` sort only the first front and exit. 
    :Returns pareto_front: (list) A list of Pareto fronts, the first list includes nondominated pop.
    .. 

    reference: [Fortin2013] Fortin, Grenier, Parizeau, "Generalizing the improved run-time complexity algorithm for non-dominated sorting",
    Proceedings of the 15th annual conference on Genetic and evolutionary computation, 2013. 
    """
    if k == 0:
        return []

    pop=list(pop.items())
    #Separate individuals according to unique fitnesses
    unique_fits = defaultdict(list)
    unfeasible_fits = [] # useful for NSGA that needs all elements even unfeasibles
    for i, ind in enumerate(pop):
        if np.sum(tuple(ind[1][1].items())[1][1]) == 0:# add if feasible
            unique_fits[tuple(list(tuple(ind[1][1].items())[0][1]))].append(ind)
        else:
            unfeasible_fits.append(ind)
    
    #Launch the sorting algorithm
    obj = len(pop[0][1][1])-1
    fitnesses = list(unique_fits.keys())
    front = dict.fromkeys(fitnesses, 0)
    if fitnesses == []:
        return [[list(sorted(pop,key=lambda x: np.sum(tuple(x[1][1].items())[1][1])))[0]],list(sorted(pop,key=lambda x: np.sum(tuple(x[1][1].items())[1][1])))[1:]],True
    # Sort the fitnesses lexicographically.
    fitnesses.sort(reverse=True)
    sortNDHelperA_constr(fitnesses, obj, front)
    
    #Extract pop from front list here
    nbfronts = max(front.values())+1
    pareto_fronts = [[] for i in range(nbfronts)]
    for fit in fitnesses:
        index = front[fit]
        pareto_fronts[index].extend(unique_fits[fit])
    
    # added for NSGA-III 
    unfeasible_fits = list(sorted(unfeasible_fits,key=lambda x: np.sum(tuple(x[1][1].items())[1][1])))# sort the unfeasible
    index = nbfronts - 1
    for fit in unfeasible_fits:
        pareto_fronts[index].extend([fit])

    # Keep only the fronts required to have k pop.
    if not first_front_only:
        count = 0
        for i, front in enumerate(pareto_fronts):
            count += len(front)
            if count >= k:

                return pareto_fronts[:i+1],False
        return pareto_fronts,False
    else:
        return pareto_fronts[0],False
