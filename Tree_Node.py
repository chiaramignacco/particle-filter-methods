import numpy as np

class Tree_Node:
    def __init__(self, tau, s, v, a, parent, weight=0, in_F_r0=0, t_r=None, t_obs_prec=None, W_ri=None, C=None):
        self.tau = tau
        self.sx, self.sy = s[0], s[1] 
        self.vx, self.vy = v[0], v[1]
        self.ax, self.ay = a[0], a[1]
        self.in_F_r0 = in_F_r0
        self.parent = parent
        self.weight = weight
        self.children = []
        self.t_r = t_r
        self.t_obs_prec = t_obs_prec
        self.W_ri = W_ri
        self.C = C
        
    #def add_child_node(self, tau, s, v, a, weight):
    #    child = Tree_Node(tau, s, v, a, self, weight)
    #    self.children.append(child)
    
    def add_child_node(self, child):
        self.children.append(child)
        
    def get_sx(self):
        return self.sx
    
    def get_sy(self):
        return self.sy
    
    def get_vx(self):
        return self.vx
    
    def get_vy(self):
        return self.vy
    
    def get_ax(self):
        return self.ax
    
    def get_ay(self):
        return self.ay
    
    def get_tau(self):
        return self.tau
    
    def get_parent(self):
        return self.parent
    
    def set_parent(self, new_parent):
        self.parent = new_parent
    
    def get_weight(self):
        return self.weight
    
    def set_weight(self, new_weight):
        self.weight = new_weight
    
    def get_in_F_r0(self):
        return self.in_F_r0
    
    def get_t_r(self):
        return self.t_r
    
    def get_t_obs_prec(self):
        return self.t_obs_prec
    
    def get_W_ri(self):
        return self.W_ri
    
    def set_C(self, C):
        self.C = C
        
    def get_C(self):
        return self.C
    
    def get_children(self):
        return self.children

