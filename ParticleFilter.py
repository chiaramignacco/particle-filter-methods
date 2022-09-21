import numpy as np
from scipy.stats import norm
from scipy.stats import gamma, bernoulli


class ParticleFilter:
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
        self.weights = np.ones(self.num_particles) / np.sum(np.ones(self.num_particles)) 
        self.store_weights = np.zeros([num_particles, self.n])
        
        self.particles =  {'s_x' : np.zeros([self.num_particles, self.capacity]),
                          's_y' : np.zeros([self.num_particles, self.capacity]),
                          'v_x' : np.zeros([self.num_particles, self.capacity]),
                          'v_y' : np.zeros([self.num_particles, self.capacity]),
                          'a_x' : np.zeros([self.num_particles, self.capacity]),
                          'a_y' : np.zeros([self.num_particles, self.capacity]),
                          'tau' : np.zeros([self.num_particles, self.capacity]),
                          }

        self.k = np.zeros(num_particles)
        
        capacity = 500
        self.expectation_x,  self.expectation_y = np.zeros(self.n), np.zeros(self.n)
        
        #[particles, k] = Init(particles, sigma_theta, num_particles, delta_t, s_x, s_y, v_x, v_y, a_x, a_y, tau, lam, s_x_0, #s_y_0, v_x_0, v_y_0);
        
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
        x_temp = self.get_space(self.particles['s_x'][j, self.k[j]-1], self.particles['v_x'][j, self.k[j]-1], self.particles['a_x'][j, self.k[j]-1], self.particles['tau'][j, self.k[j]-1], self.delta_t)
        y_temp = self.get_space(self.particles['s_y'][j, self.k[j]-1], self.particles['v_y'][j, self.k[j]-1], self.particles['a_y'][j, self.k[j]-1], self.particles['tau'][j, self.k[j]-1], self.delta_t)
        return x_temp, y_temp
        
    
    def get_likelihood(self, x_temp, x_obs, y_temp, y_obs):
        pobsx_given_x, pobsy_given_y = norm.pdf(x_temp, x_obs, self.sigma_obs), norm.pdf(y_temp, y_obs, self.sigma_obs) 
        return pobsx_given_x*pobsy_given_y
        
    def init_particles(self):
        
        self.k = np.random.poisson(lam=self.lam, size=self.num_particles) + 1
  
        for j in range(self.num_particles):
            
            ts = self.delta_t*np.random.uniform(0, 1, self.k[j]-1)
            ts = np.sort(ts)
    
            self.particles['s_x'][j, 0] = self.s_x_0
            self.particles['s_y'][j, 0] = self.s_y_0
            self.particles['v_x'][j, 0] = self.v_x_0
            self.particles['v_y'][j, 0] = self.v_y_0
            self.particles['a_x'][j, 0] = np.random.normal(0, self.sigma_theta)
            self.particles['a_y'][j, 0] = np.random.normal(0, self.sigma_theta)
            self.particles['tau'][j, 0] = 0
    
    
            for i in range(1, int(self.k[j])):
    
                self.particles['tau'][j, i] = ts[i-1]
                self.particles['a_x'][j, i] = np.random.normal(0, self.sigma_theta)
                self.particles['a_y'][j, i] = np.random.normal(0, self.sigma_theta)
                
                self.particles['s_x'][j, i] = self.get_space(self.particles['s_x'][j, i-1], self.particles['v_x'][j, i-1], self.particles['a_x'][j, i-1], self.particles['tau'][j, i-1], self.particles['tau'][j, i])
                self.particles['s_y'][j, i] = self.get_space(self.particles['s_y'][j, i-1], self.particles['v_y'][j, i-1], self.particles['a_y'][j, i-1], self.particles['tau'][j, i-1], self.particles['tau'][j, i])
                
                self.particles['v_x'][j, i] = self.get_velocity(self.particles['v_x'][j, i-1], self.particles['a_x'][j, i-1], self.particles['tau'][j, i-1], self.particles['tau'][j, i])
                self.particles['v_y'][j, i] = self.get_velocity(self.particles['v_y'][j, i-1], self.particles['a_y'][j, i-1], self.particles['tau'][j, i-1], self.particles['tau'][j, i])
    
    
    
    def birth_move(self, current_t):
        for j in range(self.num_particles):
            self.k[j] = int(self.k[j])
            ts = self.particles['tau'][j, self.k[j]-1] + (current_t*self.delta_t - self.particles['tau'][j, self.k[j]-1])*np.random.uniform(0, 1)
            self.k[j] = self.k[j] + 1
            #evaluate x and v in the new tau
            self.particles['tau'][j, self.k[j]-1] = ts
            self.particles['a_x'][j, self.k[j]-1] = np.random.normal(0, self.sigma_theta)
            self.particles['a_y'][j, self.k[j]-1] = np.random.normal(0, self.sigma_theta)
                
            self.particles['s_x'][j, self.k[j]-1] = self.get_space(self.particles['s_x'][j, self.k[j]-2], self.particles['v_x'][j, self.k[j]-2], self.particles['a_x'][j, self.k[j]-2], self.particles['tau'][j, self.k[j]-2], ts)
            self.particles['s_y'][j, self.k[j]-1] = self.get_space(self.particles['s_y'][j, self.k[j]-2], self.particles['v_y'][j, self.k[j]-1], self.particles['a_y'][j, self.k[j]-2], self.particles['tau'][j, self.k[j]-2], ts)
                
            self.particles['v_x'][j, self.k[j]-1] = self.get_velocity(self.particles['v_x'][j, self.k[j]-2], self.particles['a_x'][j, self.k[j]-2], self.particles['tau'][j, self.k[j]-1], ts)
            self.particles['v_y'][j, self.k[j]-1] = self.get_velocity(self.particles['v_y'][j, self.k[j]-2], self.particles['a_y'][j, self.k[j]-2], self.particles['tau'][j, self.k[j]-1], ts)

                        
    def multiple_birth_move(self, current_t, j):
        k_new = 2
        #k_new = 2*np.ones(n)
        
        #ts = particles(j,tau,k(j)) + delta_t*rand(1,k_new(j));
        ts = self.particles['tau'][j, self.k[j]-1] + ((current_t+1)*self.delta_t - self.particles['tau'][j, self.k[j]-1])*np.random.uniform(0, 1, k_new)
        ts = np.sort(ts)
            
        for i in range(k_new):
            self.particles['tau'][j, self.k[j]+i] = ts[i]
            self.particles['a_x'][j, self.k[j]+i] = np.random.normal(0, self.sigma_theta)
            self.particles['a_y'][j, self.k[j]+i] = np.random.normal(0, self.sigma_theta)
            self.particles['s_x'][j, self.k[j]+i] = self.get_space(self.particles['s_x'][j, self.k[j]+i-1], self.particles['v_x'][j, self.k[j]+i-1], self.particles['a_x'][j, self.k[j]+i-1], self.particles['tau'][j, self.k[j]+i-1], ts[i])
            self.particles['s_y'][j, self.k[j]+i] = self.get_space(self.particles['s_y'][j, self.k[j]+i-1], self.particles['v_y'][j, self.k[j]+i-1], self.particles['a_y'][j, self.k[j]+i-1], self.particles['tau'][j, self.k[j]+i-1], ts[i]) 
            
            self.particles['v_x'][j, self.k[j]+i] = self.get_velocity(self.particles['v_x'][j, self.k[j]+i-1], self.particles['a_x'][j, self.k[j]+i-1], self.particles['tau'][j, self.k[j]+i-1], ts[i])
            self.particles['v_y'][j, self.k[j]+i] = self.get_velocity(self.particles['v_y'][j, self.k[j]+i-1], self.particles['a_y'][j, self.k[j]+i-1], self.particles['tau'][j, self.k[j]+i-1], ts[i])

        self.k[j] = self.k[j] + k_new



    def resample(self, current_t):
        
        M = len(self.weights)
        
        ni = np.random.permutation(M)
        weights = self.weights[ni]
        
        inds = np.zeros(self.num_particles)
        
        weights = weights/np.sum(weights)
        cdf = np.cumsum(weights)
        
        cdf[-1] = 1
        
        p = np.linspace(np.random.uniform(0, 1)*(1/self.num_particles), 1, self.num_particles)
        picked = np.zeros(M)
                        
        j=0
        for i in range(self.num_particles):
            while (j < M) & (cdf[j] < p[i]):
                j += 1
            picked[j] += 1
        
        rind = 0
        for i in range(M):
            if picked[i] > 0:
                for j in range(int(picked[i])):
                    inds[rind] = int(ni[i])
                    rind = rind + 1
                    
        return inds.astype(int)
    
    def adjust_move(self, current_t, j):
        #new tau
        ts = np.random.normal(self.particles['tau'][j, self.k[j]-1], self.sigma_adjust)

        #evaluate x and v in the new tau
        self.particles['tau'][j, self.k[j]-1] = ts
        self.particles['a_x'][j, self.k[j]-1] = self.particles['a_x'][j, self.k[j]-1]
        self.particles['a_y'][j, self.k[j]-1] = self.particles['a_y'][j, self.k[j]-1]
        self.particles['s_x'][j, self.k[j]-1] = self.get_space(self.particles['s_x'][j, self.k[j]-2], self.particles['v_x'][j, self.k[j]-2], self.particles['a_x'][j, self.k[j]-2], self.particles['tau'][j, self.k[j]-2], ts)
        self.particles['s_y'][j, self.k[j]-1] = self.get_space(self.particles['s_y'][j, self.k[j]-2], self.particles['v_y'][j, self.k[j]-2], self.particles['a_y'][j, self.k[j]-2], self.particles['tau'][j, self.k[j]-2], ts)
        self.particles['v_x'][j, self.k[j]-1] = self.get_velocity(self.particles['v_x'][j, self.k[j]-2], self.particles['a_x'][j, self.k[j]-2], self.particles['tau'][j, self.k[j]-2], ts)
        self.particles['v_y'][j, self.k[j]-1] = self.get_velocity(self.particles['v_y'][j, self.k[j]-2], self.particles['a_y'][j, self.k[j]-2], self.particles['tau'][j, self.k[j]-2], ts) 

                        

    
    def compute_survivor_probability(self, current_t):
        p = np.zeros(self.num_particles)
        
        for j in range(self.num_particles):
            p[j] = gamma.cdf((current_t+1)*self.delta_t - self.particles['tau'][j, self.k[j]-1] , self.gamma_shape, self.gamma_scale) - gamma.cdf(0 , self.gamma_shape, self.gamma_scale)

        S = np.ones(self.num_particles) - p
        
        return S
        
        
    def algorithm(self):
        
        self.init_particles()
        for t in range(self.n):
            if t == 0: 
                for j in range(self.num_particles):
                    self.x_temp[j, t],  self.y_temp[j, t] = self.get_temporary_position(j, t)
                    
                self.weights = self.get_likelihood(self.x_temp[:, t], self.x_obs[t], self.y_temp[:, t], self.y_obs[t])
                self.weights = self.weights / np.sum(self.weights)
                self.store_weights[:, t] = self.weights
            
            else:
            
                #inds = self.resample(t)
                #self.num_unique_particles[t] = len(np.unique(inds))
                #for key in self.particles.keys():
                #    self.particles[key]= self.particles[key][inds]
                #self.k = self.k[inds]
                
                S = self.compute_survivor_probability(t)
                alpha = np.random.binomial(1, S, size = self.num_particles)
                #self.birth_move(t)
                for j in range(self.num_particles):
                    if self.k[j] == 1:
                        alpha[j] == 0
                    
                    if alpha[j] == 1:
                        self.x_temp[j, t],  self.y_temp[j, t] = self.get_temporary_position(j, t)
                    
                        prev_likelihood = self.get_likelihood(self.x_temp[j, t], self.x_obs[t], self.y_temp[j, t], self.y_obs[t]) 
                        
                        self.adjust_move(t, j)
                        self.x_temp[j, t],  self.y_temp[j, t] = self.get_temporary_position(j, t)
                    
                        curr_likelihood = self.get_likelihood(self.x_temp[j, t], self.x_obs[t], self.y_temp[j, t], self.y_obs[t])
                        
                        if (prev_likelihood * S[j] != 0):
                            self.weights[j] = (curr_likelihood * 1/2) / (prev_likelihood * S[j])
                        else:
                            self.weights[j] = (curr_likelihood * 1/2)
                    
                    else:
                        
                        self.x_temp[j, t],  self.y_temp[j, t] = self.get_temporary_position(j, t)
                        
                        prev_likelihood = self.get_likelihood(self.x_temp[j, t], self.x_obs[t], self.y_temp[j, t], self.y_obs[t])
                        
                        self.multiple_birth_move(t, j)
                        self.x_temp[j, t],  self.y_temp[j, t] = self.get_temporary_position(j, t)
                        
                        curr_likelihood = self.get_likelihood(self.x_temp[j, t], self.x_obs[t], self.y_temp[j, t], self.y_obs[t])
            
                        if ((prev_likelihood * (1 - S[j]) != 0) and ((t+1)*self.delta_t - self.particles['tau'][j, self.k[j]-1]) != 0):
                            self.weights[j] = (curr_likelihood * 1/2) / (prev_likelihood * (1 - S[j]) * (1 / ((t+1)*self.delta_t - self.particles['tau'][j, self.k[j]-1])))
                        else:
                            self.weights[j] = curr_likelihood * 1/2

                
                if (np.sum(self.weights[:]) != 0):
                    self.weights = self.weights / np.sum(self.weights)
                self.store_weights[:, t] = self.weights
                
            self.expectation_x[t] = np.matmul(self.weights, self.x_temp[:, t])
            self.expectation_y[t] = np.matmul(self.weights, self.y_temp[:, t])
                
            self.ess[t] = 1/np.sum(self.weights**2)
            if (self.ess[t] < 0.5*self.num_particles) & (t < self.n) :
                inds = self.resample(t)
                self.num_unique_particles[t] = len(np.unique(inds))
                for key in self.particles.keys():
                    self.particles[key] = self.particles[key][inds]
                self.k = self.k[inds]
                self.weights = np.ones(self.num_particles) / np.sum(np.ones(self.num_particles))                 
            
            
         
        return self.expectation_x, self.expectation_y, self.store_weights, self.x_temp, self.y_temp
            

        
        
        
    