'''
Parallel implementation of the Augmented Random Search method.
Horia Mania --- hmania@berkeley.edu
Aurelia Guy
Benjamin Recht 
'''

import parser
import time
import os
import numpy as np
import gym

import arspb.logz as logz
import ray
import arspb.utils as utils
import arspb.optimizers as optimizers
from arspb.policies import *
import socket
from arspb.shared_noise import *
import arspb.env_utils as env_utils
import random
from encoder import TransformerEncoder
import torch
from tqdm import tqdm
import arspb.trained_policies as tp
import json

import matplotlib.pyplot as plt
import pandas

##############################
#temp hack to create an envs_v2 pupper env

import os
import puppersim
import gin
from pybullet_envs.minitaur.envs_v2 import env_loader
import puppersim.data as pd

def create_pupper_env():
  CONFIG_DIR = puppersim.getPupperSimPath()+"/"
  _CONFIG_FILE = os.path.join(CONFIG_DIR, "pupper_pmtg.gin")
#  _NUM_STEPS = 10000
#  _ENV_RANDOM_SEED = 2 
   
  gin.bind_parameter("scene_base.SceneBase.data_root", pd.getDataPath()+"/")
  gin.parse_config_file(_CONFIG_FILE)
  env = env_loader.load()
  return env
  
  
##############################

@ray.remote
class Worker(object):
    """ 
    Object class for parallel rollout generation.
    """

    def __init__(self, env_seed,
                 env_name='',
                 policy_params = None,
                 deltas=None,
                 rollout_length=1000,
                 delta_std=0.01, 
                 encoder_params=None):

        # initialize OpenAI environment for each worker
        try:
          import pybullet_envs
        except:
          pass
        try:
          import tds_environments
        except:
          pass

        self.env = create_pupper_env()#gym.make(env_name)
        self.env.seed(env_seed)

        # each worker gets access to the shared noise table
        # with independent random streams for sampling
        # from the shared noise table. 
        self.deltas = SharedNoiseTable(deltas, env_seed + 7)
        self.policy_params = policy_params
        if policy_params['type'] == 'linear':
            print("LinearPolicy2")
            self.policy = LinearPolicy2(policy_params)
        elif policy_params['type'] == 'nn':
            print("FullyConnectedNeuralNetworkPolicy")
            self.policy = FullyConnectedNeuralNetworkPolicy(policy_params)
        else:
            raise NotImplementedError
            
        self.delta_std = delta_std
        self.rollout_length = rollout_length
        self.encoder_yo = TransformerEncoder(self.env.observation_space.shape[0])
        # self.encoder_yo.load_state_dict(encoder_params)

        
    def get_weights_plus_stats(self):
        """ 
        Get current policy weights and current statistics of past states.
        """
        #assert self.policy_params['type'] == 'linear'
        return self.policy.get_weights_plus_stats()
    

    def rollout(self, shift = 0., rollout_length = None, currentSimParameter = None):
        """ 
        Performs one rollout of maximum length rollout_length. 
        At each time-step it substracts shift from the reward.
        """
        
        if rollout_length is None:
            rollout_length = self.rollout_length

        total_reward = 0.
        steps = 0
        """
            reward is calculated based off of the environment's reward function
            this is currently proportional to speed - energy
            we want the reward to be based off of whether the trajectory information
            was useful for our encoder
            directives:
            - maintain a list of states visited in our trajectory
            - after env.step, add the latest observation to our list
            - then feed the list of observations to our encoder that 
            predicts out what our simulation parameter is. 
            - set the negative MSE loss between what the simulation parameter
            is and our encoder's prediction to y_t 
            - feed the list of observations minus the state just added
            to our encoder that predicts out our simulation parameter is.
            - set the negative MSE loss between what the simulation parameter
            is and our encoder's prediction to y_t-1 
            - report our reward as y_t - y_t-1 
        """
        ob = self.env.reset()
        #preprocess ob into a numpy array
        ob_np = self.policy.observation_filter(ob, update=self.policy.update_filter)
        if isinstance(ob_np, dict):
            ob_np = env_utils.flatten_observations(ob_np)
        ob_np = np.array(ob_np)
        #done with preprocessing

        #create a list of observations
        history = []
        history.append(ob_np)
        prev_pred = self.encoder_yo(history)
        prev_pred = prev_pred.squeeze(0).detach().numpy()[0]
        #prev_pred is a float
        total_reward = -1 * (prev_pred - currentSimParameter) ** 2 - shift
        history_of_history = [(history[:], currentSimParameter)]
        for i in range(rollout_length):
            action = self.policy.act(ob)
            ob, reward, done, _ = self.env.step(action)

            ob_np = self.policy.observation_filter(ob, update=self.policy.update_filter)
            if isinstance(ob_np, dict):
                ob_np = env_utils.flatten_observations(ob_np)
            ob_np = np.array(ob_np)

            history.append(ob_np)
            history_of_history.append((history[:], currentSimParameter))
            current_pred = self.encoder_yo(history)
            current_pred = current_pred.squeeze(0).detach().numpy()[0]
            current_distance = (current_pred - currentSimParameter) ** 2
            prev_distance = (prev_pred - currentSimParameter) ** 2

            reward = prev_distance - current_distance

            prev_pred = current_pred
            steps += 1
            total_reward += (reward - shift)
            if done:
                break
        return total_reward, steps, history_of_history

    def do_rollouts(self, w_policy, num_rollouts = 1, shift = 1, evaluate = False, currentSimParameter = None, encoder_params = None):
        """ 
        Generate multiple rollouts with a policy parametrized by w_policy.
        """

        rollout_rewards, deltas_idx, history_of_history = [], [], []
        steps = 0
        self.encoder_yo.load_state_dict(encoder_params)
        total_histories = []

        for i in range(num_rollouts):

            if evaluate:
                self.policy.update_weights(w_policy)
                deltas_idx.append(-1)
                
                # set to false so that evaluation rollouts are not used for updating state statistics
                self.policy.update_filter = False

                # for evaluation we do not shift the rewards (shift = 0) and we use the
                # default rollout length (1000 for the MuJoCo locomotion tasks)
                reward, r_steps, history_of_history = self.rollout(shift = 0., rollout_length = self.rollout_length, currentSimParameter=currentSimParameter)
                rollout_rewards.append(reward)
                total_histories.append(history_of_history[-1])
                
            else:
                idx, delta = self.deltas.get_delta(w_policy.size)
             
                delta = (self.delta_std * delta).reshape(w_policy.shape)
                deltas_idx.append(idx)

                # set to true so that state statistics are updated 
                self.policy.update_filter = True

                # compute reward and number of timesteps used for positive perturbation rollout
                self.policy.update_weights(w_policy + delta)
                pos_reward, pos_steps, history_of_history  = self.rollout(shift = shift, currentSimParameter=currentSimParameter)
                total_histories.append(history_of_history[-1])

                # compute reward and number of timesteps used for negative pertubation rollout
                self.policy.update_weights(w_policy - delta)
                neg_reward, neg_steps, history_of_history_1 = self.rollout(shift = shift, currentSimParameter=currentSimParameter) 
                history_of_history += history_of_history_1
                total_histories.append(history_of_history[-1])

                steps += pos_steps + neg_steps
                rollout_rewards.append([pos_reward, neg_reward])
                            
        return {'deltas_idx': deltas_idx, 'rollout_rewards': rollout_rewards, "steps" : steps, "history_of_history" : history_of_history, "total_histories" : total_histories}
    
    def stats_increment(self):
        self.policy.observation_filter.stats_increment()
        return

    def get_weights(self):
        return self.policy.get_weights()
    
    def get_filter(self):
        return self.policy.observation_filter

    def sync_filter(self, other):
        self.policy.observation_filter.sync(other)
        return

    
class ARSLearner(object):
    """ 
    Object class implementing the ARS algorithm.
    """

    def __init__(self, env_name='HalfCheetah-v1',
                 policy_params=None,
                 num_workers=32, 
                 num_deltas=320, 
                 deltas_used=320,
                 delta_std=0.01, 
                 logdir=None, 
                 rollout_length=4000,
                 step_size=0.01,
                 shift='constant zero',
                 params=None,
                 seed=123):

        logz.configure_output_dir(logdir)
        logz.save_params(params)
        try:
          import pybullet_envs
        except:
          pass
        try:
          import tds_environments
        except:
          pass

        env = create_pupper_env()#gym.make(env_name)
        
        self.timesteps = 0
        self.action_size = env.action_space.shape[0]
        self.ob_size = env.observation_space.shape[0]
        self.num_deltas = num_deltas
        self.deltas_used = deltas_used
        self.rollout_length = rollout_length
        self.step_size = step_size
        self.delta_std = delta_std
        self.logdir = logdir
        self.shift = shift
        self.params = params
        self.max_past_avg_reward = float('-inf')
        self.num_episodes_used = float('inf')

        
        # create shared table for storing noise
        print("Creating deltas table.")
        deltas_id = create_shared_noise.remote()
        self.deltas = SharedNoiseTable(ray.get(deltas_id), seed = seed + 3)
        print('Created deltas table.')

        # initialize workers with different random seeds
        print('Initializing workers.') 
        self.encoder_yoyo = TransformerEncoder(env.observation_space.shape[0])
        self.encoder_yoyo.load_state_dict(torch.load("encoder_3.pt"))
        self.encoder_optimizer = torch.optim.Adam(self.encoder_yoyo.parameters())
        self.num_workers = num_workers
        self.workers = [Worker.remote(seed + 7 * i,
                                      env_name=env_name,
                                      policy_params=policy_params,
                                      deltas=deltas_id,
                                      rollout_length=rollout_length,
                                      delta_std=delta_std) for i in range(num_workers)]


        # initialize policy 
        if policy_params['type'] == 'linear':
            print("LinearPolicy2")
            self.policy = LinearPolicy2(policy_params)
            self.w_policy = self.policy.get_weights()
        elif policy_params['type'] == 'nn':
            print("FullyConnectedNeuralNetworkPolicy")
            self.policy = FullyConnectedNeuralNetworkPolicy(policy_params)
            self.w_policy = self.policy.get_weights()
        else:
            raise NotImplementedError
            
        # initialize optimization algorithm
        self.optimizer = optimizers.SGD(self.w_policy, self.step_size)        
        print("Initialization of ARS complete.")

    def aggregate_rollouts(self, num_rollouts = None, evaluate = False, currentSimParameter = None):
        """ 
        Aggregate update step from rollouts generated in parallel.
        """

        if num_rollouts is None:
            num_deltas = self.num_deltas
        else:
            num_deltas = num_rollouts
            
        # put policy weights in the object store
        policy_id = ray.put(self.w_policy)

        t1 = time.time()
        num_rollouts = int(num_deltas / self.num_workers)
            
        # parallel generation of rollouts
        rollout_ids_one = [worker.do_rollouts.remote(policy_id,
                                                 num_rollouts = num_rollouts,
                                                 shift = self.shift,
                                                 evaluate=evaluate,
                                                 currentSimParameter=currentSimParameter, 
                                                 encoder_params = self.encoder_yoyo.state_dict()) for worker in self.workers]

        rollout_ids_two = [worker.do_rollouts.remote(policy_id,
                                                 num_rollouts = 1,
                                                 shift = self.shift,
                                                 evaluate=evaluate,
                                                 currentSimParameter=currentSimParameter,
                                                 encoder_params = self.encoder_yoyo.state_dict()) for worker in self.workers[:(num_deltas % self.num_workers)]]

        # gather results 
        results_one = ray.get(rollout_ids_one)
        results_two = ray.get(rollout_ids_two)

        rollout_rewards, deltas_idx, history, total_histories = [], [], [], []

        for result in results_one:
            if not evaluate:
                self.timesteps += result["steps"]
            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']
            history += result['history_of_history']
            total_histories += result['total_histories']

        for result in results_two:
            if not evaluate:
                self.timesteps += result["steps"]
            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']
            history += result['history_of_history']
            total_histories += result['total_histories']

        deltas_idx = np.array(deltas_idx)
        rollout_rewards = np.array(rollout_rewards, dtype = np.float64)
        
        print('Maximum reward of collected rollouts:', rollout_rewards.max())
        t2 = time.time()

        print('Time to generate rollouts:', t2 - t1)

        if evaluate:
            return rollout_rewards

        # select top performing directions if deltas_used < num_deltas
        max_rewards = np.max(rollout_rewards, axis = 1)
        if self.deltas_used > self.num_deltas:
            self.deltas_used = self.num_deltas
            
        idx = np.arange(max_rewards.size)[max_rewards >= np.percentile(max_rewards, 100*(1 - (self.deltas_used / self.num_deltas)))]
        deltas_idx = deltas_idx[idx]
        rollout_rewards = rollout_rewards[idx,:]
        
        # normalize rewards by their standard deviation
        np_std = np.std(rollout_rewards)
        if np_std>1e-6:
          rollout_rewards /= np_std

        t1 = time.time()
        # aggregate rollouts to form g_hat, the gradient used to compute SGD step
        g_hat, count = utils.batched_weighted_sum(rollout_rewards[:,0] - rollout_rewards[:,1],
                                                  (self.deltas.get(idx, self.w_policy.size)
                                                   for idx in deltas_idx),
                                                  batch_size = 500)
        g_hat /= deltas_idx.size
        t2 = time.time()
        print('time to aggregate rollouts', t2 - t1)
        return g_hat, history, total_histories, 
        

    def train_step(self, current_train_step=1, threshold=0):
        """ 
        Perform one update step of the policy weights.
        """
        filename = "puppersim/random.txt"
        f = open(filename, "w")
        currentSimParameter = random.random() * 5
        f.write(str(currentSimParameter) + "\n")
        f.close()
        
        g_hat, history, total_histories = self.aggregate_rollouts(currentSimParameter=currentSimParameter)                    
        print("Euclidean norm of update step:", np.linalg.norm(g_hat))
        if current_train_step > threshold:
            self.w_policy -= self.optimizer._compute_step(g_hat).reshape(self.w_policy.shape)
            print("policy updated")
        return history, total_histories

    def train(self, num_iter):

        start = time.time()
        best_mean_rewards = -1e30
        history_buffer = []
        total_histories_buffer = []
        num_encoder_training_steps = 40
        losses_report = []
        total_encoder_training_steps = 0
        average_rewards = []
        for i in range(num_iter):
            t1 = time.time()
            history, total_histories = self.train_step()
            history_buffer += history
            # random.shuffle(history_buffer)
            total_histories_buffer += total_histories
            t2 = time.time()
            print('total time of one step', t2 - t1)           
            print('iter ', i,' done')

            if (((i + 1) % 100) == 0 and i > 500):
                print('training encoder...')
                total_loss = 0
                target, pred = None, None
                for step in tqdm(range(num_encoder_training_steps)):
                    training_sample = random.sample(history_buffer, 16)
                    self.encoder_optimizer.zero_grad()
                    loss = []

                    for sample in training_sample:
                        trajectory, target = sample
                        pred = self.encoder_yoyo(trajectory)
                        pred = torch.squeeze(pred, 0)
                        loss.append((pred - target) ** 2)
                    loss = torch.stack(loss)
                    loss = torch.mean(loss, 0)
                    loss.backward()
                    total_loss += loss.item()
                    self.encoder_optimizer.step()
                    losses_report.append(loss.item())


                print('Sample simulation parameter: ', target)
                print('Sample simulation prediction: ', pred.detach().numpy()[0])
                print('Encoder training loss: ', total_loss / (num_encoder_training_steps))
                #losses_report += [total_loss / num_encoder_training_steps for ]
                plt.figure(i)
                plt.plot(np.linspace(0, len(losses_report), len(losses_report)), losses_report)
                plt.savefig("visualize.png")
                print('Distance: ', np.sqrt(total_loss / num_encoder_training_steps))
                torch.save(self.encoder_yoyo.state_dict(), "encoder.pt")

            # record statistics every 10 iterations
            if ((i + 1) % 10 == 0):
                history = []
                filename = "random.txt"
                f = open(filename, "w")
                currentSimParameter = random.random() * 0.1
                f.write(str(currentSimParameter) + "\n")
                f.close()

                f = open("output.csv", "a")
                df = pandas.DataFrame(total_histories_buffer)
                df.to_csv("output.csv",index=False)
                f.close()
                total_histories_buffer = []

                rewards = self.aggregate_rollouts(num_rollouts = 100, evaluate = True, currentSimParameter = currentSimParameter)
                w = ray.get(self.workers[0].get_weights_plus_stats.remote())
                np.savez(self.logdir + "/lin_policy_plus_latest", w)
                
                mean_rewards = np.mean(rewards)
                average_rewards.append(mean_rewards)

                if (mean_rewards > best_mean_rewards):
                  best_mean_rewards = mean_rewards
                  np.savez(self.logdir + "/lin_policy_plus_best_"+str(i+1), w)
                  
                
                print(sorted(self.params.items()))
                plt.figure(i+0.1)
                plt.plot(np.linspace(0, len(average_rewards), len(average_rewards)), average_rewards)
                plt.savefig("reward_visualization.png")
                logz.log_tabular("Time", time.time() - start)
                logz.log_tabular("Iteration", i + 1)
                logz.log_tabular("AverageReward", np.mean(rewards))
                logz.log_tabular("StdRewards", np.std(rewards))
                logz.log_tabular("MaxRewardRollout", np.max(rewards))
                logz.log_tabular("MinRewardRollout", np.min(rewards))
                logz.log_tabular("timesteps", self.timesteps)
                logz.dump_tabular()
                
            t1 = time.time()
            # get statistics from all workers
            for j in range(self.num_workers):
                self.policy.observation_filter.update(ray.get(self.workers[j].get_filter.remote()))
            self.policy.observation_filter.stats_increment()

            # make sure master filter buffer is clear
            self.policy.observation_filter.clear_buffer()
            # sync all workers
            filter_id = ray.put(self.policy.observation_filter)
            setting_filters_ids = [worker.sync_filter.remote(filter_id) for worker in self.workers]
            # waiting for sync of all workers
            ray.get(setting_filters_ids)
         
            increment_filters_ids = [worker.stats_increment.remote() for worker in self.workers]
            # waiting for increment of all workers
            ray.get(increment_filters_ids)            
            t2 = time.time()
            print('Time to sync statistics:', t2 - t1)
                        
        return 

def run_ars(params_, args):
    dir_path = params_['dir_path']
    
    if not(os.path.exists(dir_path)):
        os.makedirs(dir_path)
    logdir = dir_path
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    try:
      import pybullet_envs
    except:
      pass
    try:
      import tds_environments
    except:
      pass


    print('loading and building expert policy')
    # if len(args.json_file)==0:
    #   args.json_file = tp.getDataPath()+"/"+ args.envname+"/params.json"    
    with open(args.json_file) as f:
       params = json.load(f)
    print("params=",params)
    if len(args.expert_policy_file)==0:
      args.expert_policy_file=tp.getDataPath()+"/"+args.envname+"/nn_policy_plus.npz" 
      if not os.path.exists(args.expert_policy_file):
        args.expert_policy_file=tp.getDataPath()+"/"+args.envname+"/lin_policy_plus.npz"
    data = np.load(args.expert_policy_file, allow_pickle=True)

    print('create gym environment:', params["env_name"])
    # env = create_pupper_env(args)#gym.make(params["env_name"])
    env = create_pupper_env()#gym.make(params["env_name"])

    lst = data.files
    weights = data[lst[0]][0]
    mu = data[lst[0]][1]
    print("mu=",mu)
    std = data[lst[0]][2]
    print("std=",std)
        
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]
    ac_lb = env.action_space.low
    ac_ub = env.action_space.high
    
    policy_params={'type': params["policy_type"],
                   'ob_filter':params['filter'],
                   'ob_dim':ob_dim,
                   'ac_dim':ac_dim,
                   'action_lower_bound' : ac_lb,
                   'action_upper_bound' : ac_ub,
    }
    policy_params['weights'] = weights
    policy_params['observation_filter_mean'] = mu
    policy_params['observation_filter_std'] = std
    if params["policy_type"]=="nn":
      print("FullyConnectedNeuralNetworkPolicy")
      policy_sizes_string = params['policy_network_size_list'].split(',')
      print("policy_sizes_string=",policy_sizes_string)
      policy_sizes_list = [int(item) for item in policy_sizes_string]
      print("policy_sizes_list=",policy_sizes_list)
      policy_params['policy_network_size'] = policy_sizes_list
      policy = FullyConnectedNeuralNetworkPolicy(policy_params, update_filter=False)
    else:
      print("LinearPolicy2")
      policy = LinearPolicy2(policy_params, update_filter=False)
    policy.get_weights()




    # env = create_pupper_env()#gym.make(params['env_name'])
    # ob_dim = env.observation_space.shape[0]
    # ac_dim = env.action_space.shape[0]
    # ac_lb = env.action_space.low
    # ac_ub = env.action_space.high

    # # set policy parameters. Possible filters: 'MeanStdFilter' for v2, 'NoFilter' for v1.
    # if params["policy_type"]=="nn":
    #   policy_sizes_string = params['policy_network_size_list'].split(',')
    #   print("policy_sizes_string=",policy_sizes_string)
    #   policy_sizes_list = [int(item) for item in policy_sizes_string]
    #   print("policy_sizes_list=",policy_sizes_list)
    #   activation = params['activation']
    #   policy_params={'type': params["policy_type"],
    #                  'ob_filter':params['filter'],
    #                  'policy_network_size' : policy_sizes_list,
    #                  'ob_dim':ob_dim,
    #                  'ac_dim':ac_dim,
    #                  'activation' : activation,
    #                  'action_lower_bound' : ac_lb,
    #                  'action_upper_bound' : ac_ub,
    #   }
    # else:
    #   del params['policy_network_size_list']
    #   del params['activation']
    #   policy_params={'type': params["policy_type"],
    #                  'ob_filter':params['filter'],
    #                  'ob_dim':ob_dim,
    #                  'ac_dim':ac_dim,
    #                  'action_lower_bound' : ac_lb,
    #                  'action_upper_bound' : ac_ub,
    #   }
    
    print('blah blah blah', params_['env_name'])
    ARS = ARSLearner(env_name=params_['env_name'],
                     policy_params=policy_params,
                     num_workers=params_['n_workers'], 
                     num_deltas=params_['n_directions'],
                     deltas_used=params_['deltas_used'],
                     step_size=params_['step_size'],
                     delta_std=params_['delta_std'], 
                     logdir=logdir,
                     rollout_length=params_['rollout_length'],
                     shift=params_['shift'],
                     params=params,
                     seed = params_['seed'])
    
    ARS.train(params_['n_iter'])
       
    return 


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='InvertedPendulumSwingupBulletEnv-v0')
    parser.add_argument('--n_iter', '-n', type=int, default=100000)
    parser.add_argument('--n_directions', '-nd', type=int, default=16)
    parser.add_argument('--deltas_used', '-du', type=int, default=16)
    parser.add_argument('--step_size', '-s', type=float, default=0.03)
    parser.add_argument('--delta_std', '-std', type=float, default=.03)
    parser.add_argument('--n_workers', '-e', type=int, default=18)
    parser.add_argument('--rollout_length', '-r', type=int, default=150)
    parser.add_argument('--expert_policy_file', type=str, default="")
    parser.add_argument('--json_file', type=str, default="")

    # for Swimmer-v1 and HalfCheetah-v1 use shift = 0
    # for Hopper-v1, Walker2d-v1, and Ant-v1 use shift = 1
    # for Humanoid-v1 used shift = 5
    parser.add_argument('--shift', type=float, default=0)
    parser.add_argument('--seed', type=int, default=37)
    parser.add_argument('--policy_type', type=str, help="Policy type, linear or nn (neural network)", default= 'linear')
    parser.add_argument('--dir_path', type=str, default='data')

    # for ARS V1 use filter = 'NoFilter'
    parser.add_argument('--filter', type=str, default='MeanStdFilter')
    parser.add_argument('--activation', type=str, help="Neural network policy activation function, tanh or clip", default="tanh")
    parser.add_argument('--policy_network_size', action='store', dest='policy_network_size_list', type=str, default='64,64')
    parser.add_argument('--redis_address', type=str, default=socket.gethostbyname(socket.gethostname())+':6379') 
   
    args = parser.parse_args()

    print("redis_address=", args.redis_address)
    #ray.init(address=args.redis_address)
    ray.init()

    params = vars(args)
    run_ars(params, args)

