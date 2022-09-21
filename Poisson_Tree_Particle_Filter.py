import numpy as np
from numpy.random import choice
from scipy.stats import norm
from scipy.stats import gamma, bernoulli
from Tree_Node import *

class Poisson_Tree_Particle_Filter:
    def __init__(self, s_x, s_y, x_obs, y_obs, delta_t, n, sigma_theta, sigma_obs, gamma_shape, gamma_scale, lam, s_x_0, s_y_0, v_x_0, v_y_0, num_particles):
        
        self.s_x_0, self.s_y_0 = s_x_0, s_y_0
        self.v_x_0, self.v_y_0 = v_x_0, v_y_0
        self.x_obs, self.y_obs= x_obs, y_obs 
        self.s_x, self.s_y = s_x, s_y
        
        self.delta_t = delta_t
        self.n = n
        self.lam = lam
        self.sigma_obs, self.sigma_theta = sigma_obs, sigma_theta
        self.gamma_shape, self.gamma_scale = gamma_shape, gamma_scale
        self.num_particles = num_particles
        self.sigma_adjust = self.delta_t/1000
        self.capacity = self.n*2
        self.weights_curr = []
        self.weights_new = []
        self.weights_final = []
        self.particles_curr = []
        self.particles_new = []
        self.particles_final = []
        self.nodes_set = set()
        self.lambda_zero = 100
        self.V_act = []

        self.expectation_x,  self.expectation_y = np.zeros(self.n), np.zeros(self.n)
        
        self.num_unique_particles = np.zeros(self.n)
        self.x_temp, self.y_temp = np.zeros([self.num_particles, self.n]), np.zeros([self.num_particles, self.n])
        self.ess = np.zeros(self.n)
    
    def get_space(self, s_0, v_0, a_0, t_0, t_1):
        dt = t_1 - t_0
        return s_0 + v_0*dt + (1/2)*a_0*(dt**2)
    
    def get_velocity(self, v_0, a_0, t_0, t_1):
        dt = t_1 - t_0
        return v_0 + a_0*dt
    
    def get_temporary_position(self, j, current_t):
        x_temp = self.get_space(self.particles_new[j].sx, self.particles_new[j].vx, self.particles_new[j].ax, self.particles_new[j].tau, self.delta_t)
        y_temp = self.get_space(self.particles_new[j].sy, self.particles_new[j].vy, self.particles_new[j].ay, self.particles_new[j].tau, self.delta_t)
        return x_temp, y_temp
        
    
    def get_likelihood(self, x_temp, x_obs, y_temp, y_obs):
        pobsx_given_x, pobsy_given_y = norm.pdf(x_temp, x_obs, self.sigma_obs), norm.pdf(y_temp, y_obs, self.sigma_obs) 
        return pobsx_given_x*pobsy_given_y
        
    def init_particles(self):
          for j in range(self.num_particles):
            sx, sy = self.s_x_0, self.s_y_0
            vx, vy = self.v_x_0, self.v_y_0
            ax, ay = np.random.normal(0, self.sigma_theta), np.random.normal(0, self.sigma_theta)
                        
            x_temp = self.get_space(sx, vx, ax, 1, 0)
            y_temp = self.get_space(sy, vy, ay, 1, 0)
            weight = self.get_likelihood(x_temp, self.x_obs[0], y_temp, self.y_obs[0])
            
            particle = Tree_Node(0, [sx, sy], [vx, vy], [ax, ay], None, weight)
            self.V_act.append(particle)
            self.nodes_set.add(particle)
            self.weights_curr.append(weight) 

    def sample_children_particle(self, list_indexes, list_weights):
        return choice(list_indexes, 1, p=list_weights)
    
    def get_final_particles(self):
        return self.particles_final
    
    def get_final_weights(self):
        return self.weights_final
    
    def get_nodes(self):
        return self.nodes_set()
    
    def get_sample_trajectory(self):
        print(self.particles_final, self.weights_final)
        particle = self.sample_children_particle(self.particles_final, self.weights_final)
        trajectory_x = [particle.get_sx()]
        trajectory_y = [particle.get_sy()]
        
        while parent != None:
            parent = particle.get_parent() 
            trajectory_x.insert(0, parent.get_sx())       
            trajectory_y.insert(0, parent.get_sy())
            particle = parent
            
        return trajectory_x, trajectory_y
    
    def main_algorithm(self):
        self.init_particles()
        for t in range(self.n):
            for j in range(self.num_particles):
                particle = self.particles_curr[j]
                if np.sum(self.weights_curr) != 0:
                    Lambda = self.lambda_zero / np.sum(self.weights_curr)
                else:
                    Lambda = 1
                #print(Lambda, particle.weight, t )
                num_children = np.random.poisson(Lambda * particle.weight)
                if num_children > 0:
                    for i in range(num_children):
                        tau = particle.tau + (t*self.delta_t - particle.tau)*np.random.uniform(0, 1)
                        ax, ay = np.random.normal(0, self.sigma_theta), np.random.normal(0, self.sigma_theta)
                        sx = self.get_space(particle.get_sx(), particle.get_vx(), ax, particle.get_tau(), tau)
                        sy = self.get_space(particle.get_sy(), particle.get_vy(), ay, particle.get_tau(), tau)
                        vx = self.get_velocity(particle.get_vx(), ax, particle.get_tau(), tau)
                        vy = self.get_velocity(particle.get_vy(), ay, particle.get_tau(), tau)
                        
                        x_temp = self.get_space(sx, vx, ax, t+1, tau)
                        y_temp = self.get_space(sy, vy, ay, t+1, tau)
                        weight = self.get_likelihood(x_temp, self.x_obs[t], y_temp, self.y_obs[t])
                        
                        self.weights_new.append(weight)
                      
                        particle.add_child_node(tau, [sx, sy], [vx, vy], [ax, ay], weight)
                        
                    for child in particle.children:
                        self.particles_new.append(child)
                        self.nodes_set.add(child)
            
            self.num_particles = len(self.particles_new)
            if len(self.particles_new) == 0:
                print('Stop at ', t)
                self.particles_final = self.particles_curr
                self.weights_final = self.weights_curr / np.sum(self.weights_curr)
            self.weights_curr = self.weights_new
            self.weights_new = []
            
            self.particles_curr = self.particles_new
            self.particles_new = []
            
            if t == self.n - 1:
                print('Final iteration reached')
                self.particles_final = self.particles_curr
                self.weights_final = self.weights_curr / np.sum(self.weights_curr)
       
    
    
        
            

        
        
        
    