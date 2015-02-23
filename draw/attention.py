#!/ysr/bin/env python 

from __future__ import division

import numpy as np

import theano 
import theano.tensor as T

from theano import tensor

#-----------------------------------------------------------------------------

def batched_dot(A, B):
    batch_size,  K, L = A.shape
    batch_size_, M, N = B.shape

    assert batch_size == batch_size_
    assert L == M
    
    C = np.empty([batch_size, K, N])
    for i in xrange(batch_size):
        C[i,:,:] = np.dot(A[i], B[i])
    return C

#-----------------------------------------------------------------------------

class Reader(object):
    def __init__(self, size_y, size_x, N):
        self.size_y = size_y
        self.size_x = size_x
        self.N = N

    def read(image, gx, gy, delta, sigma):
        batch_size = image.shape[0]

        image = image.reshape( (batch_size, self.size_y, self.size_x) )

        gx = tensor.shape_padright(gx)
        gy = tensor.shape_padright(gy)
        delta = tensor.shape_padright(delta)

        muX = gx + delta * (tensor.arange(N) - N/2 - 0.5)
        muY = gy + delta * (tensor.arange(N) - N/2 - 0.5)
        
        a = tensor.arange(self.size_x)
        b = tensor.arange(self.size_y)

        muX = tensor.shape_padright(muX)
        muY = tensor.shape_padright(muY)
        
        sigma = tensor.shape_padright(sigma, 2)
        FX = tensor.exp( -(a-muX)**2 / 2. / sigma**2 )
        FY = tensor.exp( -(a-muY)**2 / 2. / sigma**2 )

        W = theano.batched_dot(theano.batched_dot(FY, image), FX.transpose([0,2,1]))
        return W.reshape((batch_size, N*N))

    #def write(W, gx, gy, delta, sigma):
    #    N = self.N
    #    
    #    batch_size, N2 = W.shape
    #    W = W.reshape( (batch_size, N, N) )
        
def read(I, N, gx, gy, delta, sigma):
    batch_size, sY, sX = I.shape

    muX = gx[:,None] + delta[:,None]*(np.arange(N)[None,:] - N/2. - .5)
    muY = gy[:,None] + delta[:,None]*(np.arange(N)[None,:] - N/2. - .5)
    
    a = np.arange(sX)
    b = np.arange(sY)

    FX = np.exp( - (a[None, None,:] - muX[:,:,None])**2 / 2. / sigma[:,None,None]**2) 
    FY = np.exp( - (b[None, None,:] - muY[:,:,None])**2 / 2. / sigma[:,None,None]**2)
    FX /= FX.sum(axis=[-1, -2])[:, None, None]
    FY /= FY.sum(axis=[-1, -2])[:, None, None]
    
    return batched_dot(batched_dot(FY, I), FX.transpose([0,2,1]))

def write(C, W, gx, gy, delta, sigma):
    N, _ = W.shape
    sY, sX = C.shape

    muX = gx + delta*(np.arange(N) - N/2. - .5)
    muY = gy + delta*(np.arange(N) - N/2. - .5)
    
    a = np.arange(sX)
    b = np.arange(sY)

    FX = np.exp( - (a[None,:] - muX[:,None])**2 / 2. / sigma**2) 
    FY = np.exp( - (b[None,:] - muY[:,None])**2 / 2. / sigma**2)
    FX /= FX.sum()
    FY /= FX.sum()
 
    C += np.dot(np.dot(FY.T, W), FX)
    return C

if __name__ == "__main__":
    import Image

    I = Image.open("cat.jpg")
    I = I.resize((640, 480)).convert('L')
    
    i = np.asarray(I)
    gx = 320
    gy = 240

    N = 40
    delta = 2
    sigma = 2

    
