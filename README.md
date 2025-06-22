# PEARL-MOO
 **P**areto **E**nvelope **A**ugmented with **R**einforcement **L**earning for **M**ulti-**O**bjective **O**ptimization (PEARL-MOO) is a python package that is intended to solve challenges relevant to operation & optimization research, engineering, business, and other disciplines that are multi-objective by nature. It supports minimization and maximization, as well as unconstrained and constrained design optimization with integer and continuous input spaces. We encourage any automated optimization practicioner and researcher around the world to contribute, extend, and utilize this framework for solving large-scale engineering design problems, in particular when a factored representation of the objective function is not known.
# Features
  * **Pareto Envelope Augmented with Reinforcement Learning (PEARL)**: Pareto Envelope Augmented with Reinforcement Learning (PEARL) is a multi-objective optimization technique based on reinforcement learning developed for efficient multi-purpose engineering design. It is compatible with Proximal Policy Optimization (PPO), Asynchronous asynchronous Advantage Actor-Critic (A2C), and Actor-Critic using Kronecker-factored Trust Region (ACKTR). See the original paper [here](https://www.sciencedirect.com/science/article/pii/S0306454924002457).
  * **Classical multiobjective heuristics-based optimization**:
  * * **Non-dominated Sorting Genetic Algorithms II and III (NSGA)**: The implemention and adaption to this package draw inspiration form our collagues at [DEAP](https://deap.readthedocs.io/en/master/). See the original papers for [NSGAII](https://www.scopus.com/record/display.uri?eid=2-s2.0-0036530772&origin=inward) and [NSGAIII](https://ieeexplore.ieee.org/document/6600851).
  * * **Constrained Non-dominated Sorting Genetic Algorithm III**: Extension of NSGA-III to deal with constrained following the seminal [paper](https://www.scopus.com/record/display.uri?eid=2-s2.0-84905579836&origin=inward).
  * **Wrapped around Neuro-Evolution Optimizaiton with Reinforcement Learning (NEORL) package**: PEARL-MOO is wrapped around NEORL, a single-objective optimization package  benefitting from a user-friendly API. This characteristics enables seamless integration of NEORL's function and capabilities including automated hyper-parameter tuning and parallel processing.
  * **Parallel Processing**: Built-in support for parallel evaluation of objective functions: In PEARL-MOO, each agent is associated a processor allowing MC rollout in parallel. For the classical NSGAII/III, each member of the population is evaluated in parallel before the seleciton process.
 * **Callback system**: Callback mechanism for monitoring and controlling the optimization process.
 * **Comprehensive set of examples**: Tutorial that demonstrates the use of pearlmoo with a suit of constrained and unconstrained classical optimiztions problems from litterature including the [dtlzX](https://ieeexplore.ieee.org/document/1007032), [cXdtlzY](https://www.scopus.com/record/display.uri?eid=2-s2.0-84905579836&origin=inward), and [ctpX](https://dl.acm.org/doi/10.5555/647889.736526) test suits.

# Installation
PEARL-MOO only requires NEORL's dependency and the python multiprocessing package pathos.

For NEORL please see the installation's instructions [here](https://neorl.readthedocs.io/en/latest/guide/detinstall.html). For pathos use pip:
```
pip install pathos
```

# Quick Start
Here's a simple example of how to use PEARL-MOO with one of the PEARL algorithms. First import the libraries
```python
from pearlmoo.utils.tools import uniform_reference_points # used to generate approprietaly spaced pareto front
from pearlmoo.callbacks.pearlmoocalls
# Import PEARL
from pearlmoo.methods.pearl import CreateEnvironment

```
Then, import the multi-objectiveproblem
```python
# Generate a multi-objective optimization problem from pymop
from pymop.factory import get_problem
PROBLEM = 'c3dtlz4'
NOBJ = 3
K = 5
NDIM = NOBJ + K - 1
P = 12
H = factorial(NOBJ + P - 1) / (factorial(P) * factorial(NOBJ - 1))
MU = int(H + (4 - H % 4))
lambda_ = MU
# create the reference directions to be used for the optimization
problem = get_problem(PROBLEM, n_var=NDIM, n_obj=NOBJ)
ref_points = uniform_reference_points(nobj = NOBJ, p = P)
pf = problem.pareto_front(ref_points)
# Generate bounds
BOUND_LOW, BOUND_UP = 0.0,1.0
nx=NDIM
BOUNDS={}
for i in range(1,nx+1):
    BOUNDS['x'+str(i)]=['float', BOUND_LOW, BOUND_UP]

def fitness_wrapper(*args, **kwargs):
    """
    Modify classical output of the form F,CV to dict read by constrained multi-objective versions
    {'objectives'F:,'constraints':CV}
    """
    fitness = problem.evaluate(*args, **kwargs)# objective function
    return {"objectives":fitness[0],"constraints":fitness[1]}# the optimizer accept a dictionary
```
Create the environment for optimization and perform it.
```python
# Create PEARL environment
ncores = 8
env=CreateEnvironment(method='ppo', fit=fitness_wrapper,nadir=[3,3,3],cl_penalty=92,buffer_size = 92,
    bounds=BOUNDS, mode='min',paradigm="constrained",ncores=ncores,selection='niching',sorting="standard",ref_points = ref_points)
# Instantiate RL agents, use default hyper-parameter for now
ppo = PPO2(MlpPolicy, env=env, n_steps=32, ent_coef= 0.0001,gamma=0.99, learning_rate=0.0025, vf_coef=0.5, max_grad_norm=0.5, lam=0.95, nminibatches=4, noptepochs=10, cliprange=0.2)                      
#create a callback function to log data
cb=PEARLRLLogger(check_freq=5,verbose=True)
# Perform the optimization
ppo.learn(total_timesteps=10000, callback=cb)
                             
```
Post-process the results
```python
W,Y,Z,X,output = [],[],[],[],[]
buffer_pop = {}
for data in cb.pearl_hist['global_fitness']:
    buffer_pop[len(buffer_pop) + 1] = data
    X.append(data)
            
pearl_sol = []
for elem in [-np.array(x) for x in buffer_pop.values()]:
    isdominated = False
    for elem2 in pf:# Compare the elements generated to a known pareto front
        if isDominated(-np.array(elem),-elem2):
            isdominated = True
    if not isdominated:
        pearl_sol.append(elem)
    W.append(elem[0])
    Y.append(elem[1])
    Z.append(elem[2])


print("--- :")
print("Pareto Optimal solutions:",len(pearl_sol))
pearl_sol = np.unique(pearl_sol,axis=0)
print('Unique Pareto Optimal solutions',pearl_sol.shape)

if buffer_pop != {}:
    nadir = np.max([[-np.array(x) for x in buffer_pop.values()]],axis=1)[0]
    print("nadir",nadir)
    nadir_ref = [3,3,3]
    print('hyper volume PEARL ',hypervolume([tuple(x) for x in [-np.array(x) for x in buffer_pop.values()]], np.array(nadir_ref)))

# Plot the final pareto front versus the pre-generated one
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(W,Y,Z, marker = '*',color = 'red')
ax.scatter(np.array(pf)[:,0],np.array(pf)[:,1],np.array(pf)[:,2], marker = 'x',color = 'black')
ax.view_init(elev=30, azim=45)
ax.set_xlabel('f1')
ax.set_ylabel('f2')
ax.set_zlabel('f3')
```
```
--- :
Pareto Optimal solutions: 224
Unique Pareto Optimal solutions (224, 3)
nadir [1.06958054 1.048875   1.15240413]
hyper volume PEARL  26.316084582716737
```
![image](https://media.github.inl.gov/user/1229/files/eb515c82-592a-4981-9802-2c03779ad656)![image](https://media.github.inl.gov/user/1229/files/e0d69363-8705-4256-9a20-0a8f0aab01d2)

# Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

# Examples of works using PEARL
Contact us if you want to get added!
```
@article{seurin2024physics,
author={Seurin, P. and Shirvan, K.},
title={Physics-Informed Reinforcement Learning Optimization of Pwr Core Loading Pattern},
journal={Annals of Nuclear Energy},
Volume={208},
year={2024},
page={110763},
ISSN={0306-4549},
url={https://doi.org/10.1016/j.anucene.2024.110763}
}
```

```
@article{seurin2025impact,
author={Seurin, P. and Halimi, A. and Shirvan, K.},
title={Impact of including fuel performance as part of core reload optimization: Application to power uprates},
journal={Nuclear Engineering and Design},
Volume={433},
year={2025},
pages={113844},
ISSN={0029-5493},
url={https://doi.org/10.1016/j.nucengdes.2025.113844.}}
```

# Citation
If you use the algorithms, please cite the seminal work
```
@article{seurin2024multiobjective,
author={Seurin, P. and Shirvan, K.},
title={Multi-objective reinforcement learning-based approach for pressurized water reactor optimization},
journal={Annals of Nuclear Energy},
Volume={205},
year={2024},
pages={110582},
doi={https://doi.org/10.1016/j.anucene.2024.110582}
}

```
and NEORL

```
@article{radaideh2023NEORL,
author={Radaideh, M. I. and Seurin, P.  and Du, K. and Seyler, D. and Gu, X. and Wang, H. and Shirvan, K.},
title ={NEORL: NeuroEvolution Optimization with Reinforcement Learning—Applications to carbon-free energy systems},
year={2023},
pages={112423},
doi = {https://doi.org/10.1016/j.nucengdes.2023.112423.},
journal={Nuclear Engineering and Design}
}
```

# Contact
email: paseurin@alum.mit.edu, paul.seurin@inl.gov

GitHub Issues: [Open an issue](https://github.com/pseur/PEARL-MOO/issues)
