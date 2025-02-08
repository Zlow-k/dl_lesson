import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self,params, grads):
        for i in range(len(params)):
            params[i] -= self.lr*grads[i]
    
            
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = []
            for param in params:
                self.v.append(np.zeros_like(param))
                
        for i in range(len(params)):
            self.v[i] = self.momentum * self.v[i] - self.lr * grads[i]
            params[i] += self.v[i]


class Nesterov:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = []
            for param in params:
                self.v.append(np.zeros_like(param))
                
        for i in range(len(params)):
            self.v[i] *= self.momentum
            self.v[i] -= self.lr * grads[i]
            params[i] += self.momentum * self.momentum * self.v[i]
            params[i] -= (1 + self.momentum) * self.lr * grads[i] 
        
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
        
    def update(self, params, grads):
        if self.h in None:
            self.h = []
            for param in params:
                self.h.append(np.zeros_like(param))
        
        for i in range(len(params)):
            self.h[i] += grads[i] * grads[i]
            params[i] -= self.lr * grads[i] /(np.sqrt(self.h[i] + 1e-7))
   
   
class RMSprop:
    def __init__(self, lr=0.01, decray_rate = 0.99):
        self.lr = lr
        self.decray_rate = decray_rate
        self.h = None
        
    def update(self, params, grads):
        if self.h in None:
            self.h = []
            for param in params:
                self.h.append(np.zeros_like(param))
                
        for i in range(len(params)):
            self.h[i] *= self.decray_rate
            self.h[i] += (1 - self.decray_rate) *grads[i] * grads[i]
            param[i] -= self.lr* grads[i] / (np.sqrt(self.h[i] + 1e-7))
     
            
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m in None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))
        self.iter += 1
        
        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i]**2
            _m = self.m[i] / (1 - self.beta1**self.iter)
            _v = self.v[i] / (1 - self.beta2**self.iter)
            
            params[i] -= self.lr * _m / (np.sqrt(_v) + 1e-7)