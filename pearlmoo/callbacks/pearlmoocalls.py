#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#"""
#@author: paul
#"""

import numpy as np
import pandas as pd
from neorl.rl.baselines.shared.callbacks import BaseCallback
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import copy
from pearlmoo.utils.hyper_volume import hypervolume
import itertools
class SavePlotCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).
    """
    def __init__(self, check_freq, avg_step, log_dir, total_timesteps, basecall, plot_mode='subplot'):
        self.base=basecall
        self.plot_mode=plot_mode
        self.n_calls=self.base.n_calls
        self.model=self.base.model
        self.num_timesteps=self.base.num_timesteps
        self.total_timesteps=total_timesteps
        self.verbose=1
        self.check_freq = check_freq
        self.avg_step=avg_step
        self.log_dir = log_dir
        self.best_save_path = self.log_dir + '_bestmodel.pkl'
        self.save_path = self.log_dir + '_lastmodel.pkl'
        self.best_mean_reward = -np.inf

        #avoid activating 'Agg' in the header so not to affect other classes/algs
        import matplotlib
        matplotlib.use('Agg')

    def runcall(self):
        
        print('num_timesteps={}/{}'.format (self.num_timesteps, self.total_timesteps))
            
        # Retrieve training reward
        y= pd.read_csv(self.log_dir+'_out.csv')
        y=y["reward"].values
        # Mean training reward over the last 100 episodes
        mean_reward = np.mean(y[-self.avg_step:])
               
        # New best model, you could save the agent here
        print('--debug: current mean reward={}, previous best mean reward = {}'.format(np.round(mean_reward), np.round(self.best_mean_reward)))
        if mean_reward > self.best_mean_reward:
              self.best_mean_reward = copy.copy(mean_reward)
              #saving best model
              print('--debug: improvement in reward is observed, new best model is saved to {}'.format(self.best_save_path))
              self.model.save(self.best_save_path)    #best model found so far

        #saving current model
        print('--debug: current model model is saved to {}'.format(self.save_path))
        self.model.save(self.save_path)   #latest model
              
        self.out_data=pd.read_csv(self.log_dir+'_out.csv')
        #-------------------
        # Progress Plot
        #-------------------
        self.plot_progress()
                
    def _on_step(self) -> bool:
        
        try:
            if (self.num_timesteps % self.check_freq == 0) or (self.num_timesteps == self.total_timesteps):
                self.runcall()
        except:
            print('--warning: No plot is generated, NEORL tried to plot the output csv logger, but failed for some reason, you may increase `check_freq` to a large value to allow some data printed in the csv logger')
        
        if self.num_timesteps == self.total_timesteps:
            print('system exit')
            os._exit(1)
            
            
        return True
    
    def _on_training_end(self) -> None:
        self.runcall()
        print('Training is finished')
        os._exit(1)
        #pass

    def calc_cumavg(self, data, N):
    
        cum_aves=[np.mean(data[i:i+N]) for i in range(0,len(data),N)]
        cum_std=[np.std(data[i:i+N]) for i in range(0,len(data),N)]
        cum_max=[np.max(data[i:i+N]) for i in range(0,len(data),N)]
        cum_min=[np.min(data[i:i+N]) for i in range(0,len(data),N)]
    
        return cum_aves, cum_std, cum_max, cum_min
    
    
    def plot_progress(self, method_xlabel='Epoch'):

        self.out_data=pd.read_csv(self.log_dir+'_out.csv')
        color_list=['b', 'g', 'r', 'c', 'm', 'y', 'darkorange', 'purple', 'tab:brown', 'lime']
        plot_data=self.out_data.drop(['caseid'], axis=1)  #exclude caseid, which is the first column from plotting (meaningless)
        
        labels=list(plot_data.columns.values)
            
        ny=plot_data.shape[1] 
        
        assert ny == len(labels), 'number of columns ({}) to plot in the csv file {} is not equal to the number of labels provided by the user ({})'.format(ny, self.log_dir+'_out.csv', len(labels))
        
        # classic mode
        if self.plot_mode=='classic' or ny == 1:
            color_index=0
            for i in range (ny): #exclude caseid from plot, which is the first column 
                plt.figure()
                ravg, rstd, rmax, rmin=self.calc_cumavg(plot_data.iloc[:,i],self.avg_step)
                epochs=np.array(range(1,len(ravg)+1),dtype=int)
                plt.plot(epochs, ravg,'-o', c=color_list[color_index], label='Average per {}'.format(method_xlabel))
                
                plt.fill_between(epochs,[a_i - b_i for a_i, b_i in zip(ravg, rstd)], [a_i + b_i for a_i, b_i in zip(ravg, rstd)],
                alpha=0.2, edgecolor=color_list[color_index], facecolor=color_list[color_index], label=r'$1-\sigma$ per {}'.format(method_xlabel))
                
                plt.plot(epochs, rmax,'s', c='k', label='Max per {}'.format(method_xlabel), markersize=4)
                plt.plot(epochs,rmin,'d', c='k', label='Min per {}'.format(method_xlabel), markersize=4)
                plt.legend()
                plt.xlabel(method_xlabel)
                plt.ylabel(labels[i])
                
                if color_index==9:
                    color_index=0
                else:
                    color_index+=1
                    
                plt.tight_layout()
                plt.savefig(self.log_dir+'_'+labels[i]+'.png', format='png', dpi=150)
                plt.close()
        
        # subplot mode           
        elif self.plot_mode=='subplot':
            # determine subplot size
            if ny == 2:
                xx= [(1,2,1),(1,2,2)]
                plt.figure(figsize=(12, 4.0))
            elif ny==3:
                xx= [(1,3,1), (1,3,2), (1,3,3)]
                plt.figure(figsize=(12, 4.0))
            elif ny==4:
                xx= [(2,2,1), (2,2,2), (2,2,3), (2,2,4)]
                plt.figure(figsize=(12, 8))
            elif ny > 4 and ny <= 21:
                nrows=int(np.ceil(ny/3))
                xx= [(nrows,3,item) for item in range(1,ny+1)]
                adj_fac=(nrows - 2.0)*0.25 + 1
                plt.figure(figsize=(12, adj_fac*8))
            elif ny > 21 and ny <= 99:
                nrows=int(np.ceil(ny/4))
                xx= [(nrows,4,item) for item in range(1,ny+1)]
                adj_fac=(nrows - 2.0)*0.25 + 1
                plt.figure(figsize=(15, adj_fac*8))
                
                
            color_index=0
            for i in range (ny): #exclude caseid from plot, which is the first column 
                plt.subplot(xx[i][0], xx[i][1], xx[i][2])
                ravg, rstd, rmax, rmin=self.calc_cumavg(plot_data.iloc[:,i],self.avg_step)
                epochs=np.array(range(1,len(ravg)+1),dtype=int)
                plt.plot(epochs,ravg,'-o', c=color_list[color_index])
                
                plt.fill_between(epochs,[a_i - b_i for a_i, b_i in zip(ravg, rstd)], [a_i + b_i for a_i, b_i in zip(ravg, rstd)],
                alpha=0.2, edgecolor=color_list[color_index], facecolor=color_list[color_index])
                
                plt.plot(epochs,rmax,'s', c='k', markersize=4)
                
                plt.plot(epochs,rmin,'d', c='k', markersize=4)
                plt.xlabel(method_xlabel)
                plt.ylabel(labels[i])
                if color_index==9:
                    color_index=0
                else:
                    color_index+=1
            
            #speical legend is created for all subplots to save space
            legend_elements = [Line2D([0], [0], color='k', marker='o', label='Mean ' + r'$\pm$ ' +r'$1\sigma$' + ' per {} (color changes)'.format(method_xlabel)),
                  Line2D([0], [0], color='k', marker='s', label='Max per {} (color changes)'.format(method_xlabel)),
                  Line2D([0], [0], linestyle='-.', color='k', marker='d', label='Min per {} (color changes)'.format(method_xlabel))]
            plt.figlegend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3)
            plt.tight_layout()
            plt.savefig(self.log_dir+'_res.png', format='png', dpi=200, bbox_inches="tight")
            plt.close()
            
        else:
            raise Exception ('the plot mode defined by the user does not exist')    

class PEARLRLLogger(BaseCallback):
    """
    Callback for logging data of RL algorithms (x,y), compatible with: A2C, ACKTR, PPO, FNEAT, RNEAT

    :param check_freq: (int) logging frequency, e.g. 1 will record every time step 
    :param plot_freq: (int) frequency of plotting the fitness progress (if ``None``, plotter is deactivated)
    :param n_avg_steps: (int) if ``plot_freq`` is NOT ``None``, then this is the number of timesteps to group to draw statistics for the plotter (e.g. 10 will group every 10 time steps to estimate min, max, mean, and std).
    :param pngname: (str) name of the plot that will be saved if ``plot_freq`` is NOT ``None``.
    :param save_model: (bool) whether or not to save the RL neural network model (model is saved every ``check_freq``)
    :param model_name: (str) name of the model to be saved  if ``save_model=True``
    :param save_best_only: (bool) if ``save_model = True``, then this flag only saves the model if the fitness value improves. 
    :param verbose: (bool) print updates to the screen
    """
    def __init__(self, check_freq=1, plot_freq=None, n_avg_steps=10, pngname='history', 
                 save_model=False, model_name='bestmodel.pkl', save_best_only=True, 
                 verbose=False):
        super(PEARLRLLogger, self).__init__(verbose)
        self.check_freq = check_freq
        self.plot_freq=plot_freq
        self.pngname=pngname
        self.n_avg_steps=n_avg_steps
        self.model_name = model_name
        self.save_model=save_model
        self.verbose=verbose
        self.save_best_only=save_best_only
        self.rbest = -np.inf
        self.rbest_maxonly = -np.inf
        self.r_hist=[]
        self.x_hist=[]
        
        self.pearl_hist = {}
        self.pearl_hist['local_fitness'] = [] # record the pareto front each self.ncalls%check_freq
        if self.plot_freq:
            #avoid activating 'Agg' in the header so not to affect other classes/algs
            import matplotlib
            matplotlib.use('Agg')
            
    def _init_callback(self) -> None:
        # Create folder if needed
        try:
            self.mode=self.training_env.get_attr('mode')[0]   #PPO/ACER/A2C/ACKTR
        except:
            try:
                self.mode=self.training_env.mode       #DQN
            except:
                print('--warning: the logger cannot find mode in the environment, it is set by default to `max`')
                self.mode='max'
        if self.mode not in ['min', 'max']:
            self.mode='max'
            print('--warning: The mode entered by user is invalid, use either `min` or `max`')

        self.Nadir =  self.training_env.get_attr('nadir')[0]# Nadir utilize to evaluate the HV
        if self.Nadir is None:
            if self.mode in ['min']:
                self.Nadir = [1e12 for i in range(self.training_env.get_attr('n_obj')[0])]
            elif self.mode in [ 'max']:
                self.Nadir = [-1e12 for i in range(self.training_env.get_attr('n_obj')[0])]
            print('--warning: The user did not provide a nadir. Arbitrary large values will be utilized to evaluate the hyper-volume in the callback function.')

    def _on_step(self) -> bool:

        if self.n_calls % self.check_freq == 0:
            if self.verbose:
                print('----------------------------------------------------------------------------------')
                print('RL callback at step {}/{}'.format(self.n_calls*len(self.locals['rewards']), self.locals['total_timesteps']))
            
            # Compute Hyper-Volume of the solutions found
            # If no improvement, no need to save
            Solutions = [x['global_fitness'] for x in self.training_env.get_attr('pearl_hist')]
            Solutions = list(itertools.chain.from_iterable(Solutions))
            
            if self.training_env.get_attr('paradigm')[0] == 'constrained': # In this instance, only feasible solutions are stored for crowding/niching techniques.
                Solutions = [x[:-1] for x in Solutions if x[-1] == 0]
                if Solutions == []: # If the buffer is empty, no feasible solutions have been found. The value of the reward can be utilized to save the best environment
                    rwd=self.locals['rewards'][0] #
                else:
                    if self.mode in ['min']:
                        HV = hypervolume([tuple(x) for x in [-y for y in Solutions]], np.array(self.Nadir))
                    elif self.mode in [ 'max']:
                        HV = hypervolume([tuple(x) for x in Solutions], np.array(self.Nadir))
                    rwd = HV # episode length is 1 
            else:
                if self.mode in ['min']:
                    HV = hypervolume([tuple(x) for x in [-y for y in Solutions]], np.array(self.Nadir))
                elif self.mode in [ 'max']:
                    HV = hypervolume([tuple(x) for x in Solutions], np.array(self.Nadir))
                rwd = HV # episode length is 1 
            
            x=self.locals['infos'][0]['x'] #A2C/PPO/ACKTR/RNEAT/FNEAT cases.
                    
            if self.save_model and not self.save_best_only:
                self.model.save(self.model_name)
                if self.verbose:
                    print('A new model is saved to {}'.format(self.model_name))
            
            if rwd > self.rbest_maxonly:
                self.xbest=x.copy()
                self.rbest_maxonly=rwd
                self.rbest=self.rbest_maxonly
            
                if self.save_model and self.save_best_only:
                    self.model.save(self.model_name)
                    if self.verbose:
                        print('An improvement is observed, new model is saved to {}'.format(self.model_name))
            
            self.r_hist.append(rwd)
            self.x_hist.append(list(x))
            
            # Update Best pareto front
            # ---
            self.pearl_hist['global_fitness'] = Solutions

            # Update history of Pareto Front solutions
            self.pearl_hist['local_fitness'].append(Solutions)
            if self.mode=='max':
                self.pearl_hist['utopia'] = list(map(max,zip(*self.pearl_hist['global_fitness'] )))
                self.pearl_hist['nadir'] = list(map(min,zip(*self.pearl_hist['global_fitness'] )))
            else:
                self.pearl_hist['utopia'] = list(map(min,zip(*[-1*x for x in self.pearl_hist['global_fitness'] ])))
                self.pearl_hist['nadir'] = list(map(max,zip(*[-1*x for x in self.pearl_hist['global_fitness'] ])))
        
            
            self.pearl_hist['objective'] = [x['objective'] for x in self.training_env.get_attr('pearl_hist')]
            # History of objectives
            if self.plot_freq:
                if self.n_calls % self.plot_freq == 0:
                    self.plot_progress()
                
            
            if self.verbose:
                print('----------------------------------------------------------------------------------')
        return True
    
    def plot_progress(self): 
    
        plt.figure()
        
        ravg, rstd, rmax, rmin=self.calc_cumavg(self.r_hist,self.n_avg_steps)
        epochs=np.array(range(1,len(ravg)+1),dtype=int)
        plt.plot(epochs, ravg,'-o', c='g', label='Average per epoch')
        
        plt.fill_between(epochs,[a_i - b_i for a_i, b_i in zip(ravg, rstd)], [a_i + b_i for a_i, b_i in zip(ravg, rstd)],
        alpha=0.2, edgecolor='g', facecolor='g', label=r'$1-\sigma$ per epoch')
        
        plt.plot(epochs, rmax,'s', c='k', label='Max per epoch', markersize=4)
        plt.plot(epochs,rmin,'d', c='k', label='Min per epoch', markersize=4)
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Fitness (HV)')
        plt.savefig(self.pngname+'.png',format='png' ,dpi=300, bbox_inches="tight")
        plt.close()

        n_obj = self.training_env.get_attr('n_obj')[0]
        fig,ax=plt.subplots(n_obj,1,figsize=(20,20))
        for i in range(n_obj):#np.array(self.pearl_hist['local_fitness'])):
            obj = [ind[i] for x in self.pearl_hist['local_fitness'] for ind in x]
            ravg, rstd, rmax, rmin=self.calc_cumavg(obj,self.n_avg_steps)
            epochs=np.array(range(1,len(ravg)+1),dtype=int)
            ax[i].plot(epochs, ravg,'-o', c='g', label='Average per epoch')
            ax[i].fill_between(epochs,[a_i - b_i for a_i, b_i in zip(ravg, rstd)], [a_i + b_i for a_i, b_i in zip(ravg, rstd)],
                alpha=0.2, edgecolor='g', facecolor='g', label=r'$1-\sigma$ per epoch')
            ax[i].plot(epochs, rmax,'s', c='k', label='Max per epoch', markersize=4)
            ax[i].plot(epochs,rmin,'d', c='k', label='Min per epoch', markersize=4)
            ax[i].set_xlabel('Epoch')
            ax[i].set_ylabel('Objective %d'%(i+1))
        plt.savefig(self.pngname+'_moo.png',format='png' ,dpi=300, bbox_inches="tight")
        plt.close()
        


    def calc_cumavg(self, data, N):
    
        cum_aves=[np.mean(data[i:i+N]) for i in range(0,len(data),N)]
        cum_std=[np.std(data[i:i+N]) for i in range(0,len(data),N)]
        cum_max=[np.max(data[i:i+N]) for i in range(0,len(data),N)]
        cum_min=[np.min(data[i:i+N]) for i in range(0,len(data),N)]
    
        return cum_aves, cum_std, cum_max, cum_min