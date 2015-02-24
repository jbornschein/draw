#!/ysr/bin/env python 

from __future__ import division

import numpy as np

import theano 
import theano.tensor as T

from theano import tensor

#-----------------------------------------------------------------------------

def batched_dot(A, B):
    return (A[:,:,:,None]*B[:,None,:,:]).sum(axis=-2)

def f_batched_dot(A, B):
    C = A.dimshuffle([0,1,2,'x']) * B.dimshuffle([0,'x',1,2])
    return C.sum(axis=-2)

def vectorize(*args):
    return [a.reshape((1,)+a.shape) for a in args]

#-----------------------------------------------------------------------------

class ZoomableAttentionWindow(object):
    """
    """
    def __init__(self, img_height, img_width, N, normalize=False):
        self.normalize = normalize
        self.img_height = img_height
        self.img_width = img_width
        self.N = N

    def filterbank_matrices(self, center_y, center_x, delta, sigma):
        """Create a Fx and a Fy
        
        Parameters
        ----------
        center_y : T.vector (shape: batch_size)
        center_x : T.vector (shape: batch_size)
            Y and X center coordinates for the attention window
        delta : T.vector (shape: batch_size)
        sigma : T.vector (shape: batch_size)
        
        Returns
        -------
            FX, FY 
        """
        N = self.N

        if self.normalize:
            center_x = center_x * self.img_width
            center_y = center_y * self.img_height
            delta = (max(self.img_width, self.img_height)-1) / (self.N-1) * delta
        
        muX = center_x.dimshuffle([0, 'x']) + delta.dimshuffle([0, 'x'])*(T.arange(N)-N/2-0.5)
        muY = center_y.dimshuffle([0, 'x']) + delta.dimshuffle([0, 'x'])*(T.arange(N)-N/2-0.5)

        a = tensor.arange(self.img_width)
        b = tensor.arange(self.img_height)
        
        FX = tensor.exp( -(a-muX.dimshuffle([0,1,'x']))**2 / 2. / sigma.dimshuffle([0,'x','x'])**2 )
        FY = tensor.exp( -(b-muY.dimshuffle([0,1,'x']))**2 / 2. / sigma.dimshuffle([0,'x','x'])**2 )
        FX = FX / FX.sum(axis=(-1, -2)).dimshuffle(0, 'x', 'x')
        FY = FY / FY.sum(axis=(-1, -2)).dimshuffle(0, 'x', 'x')

        return FX, FY


    def read(self, images, center_y, center_x, delta, sigma):
        """
        Parameters
        ----------
        image : T.matrix    (shape: batch_size x img_size)
            Batch of images. Internally it will be reshaped to be a 
            (batch_size, img_height, img_width)-shaped stack of images.
        center_y : T.vector (shape: batch_size)
        center_x : T.vector (shape: batch_size)
        delta : T.vector    (shape: batch_size)
        sigma : T.vector    (shape: batch_size)

        returns
        -------
        window : T.matrix   (shape: batch_size x N**2)
        """
        N = self.N
        batch_size = images.shape[0]

        # Reshape input into proper 2d images
        I = images.reshape( (batch_size, self.img_height, self.img_width) )

        # Get separable filterbank
        FX, FY = self.filterbank_matrices(center_y, center_x, delta, sigma)

        # Apply to the batch of images
        W = f_batched_dot(f_batched_dot(FY, I), FX.transpose([0,2,1]))

        return W.reshape((batch_size, N*N))


    def write(self, windows, center_y, center_x, delta, sigma):
        N = self.N
        batch_size = windows.shape[0]

        # Reshape input into proper 2d windows
        W = windows.reshape( (batch_size, N, N) )

        # Get separable filterbank
        FX, FY = self.filterbank_matrices(center_y, center_x, delta, sigma)

        # Apply...
        I = f_batched_dot(f_batched_dot(FY.transpose([0,2,1]), W), FX)

        return I.reshape( (batch_size, self.img_height*self.img_width) )

        """
        gx = g[:,1]
        gy = g[:,0]

        #image = image.reshape( (batch_size, self.size_y, self.size_x) )

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
        FY = tensor.exp( -(b-muY)**2 / 2. / sigma**2 )
        FX = FX / FX.sum(axis=(-1, -2)).dimshuffle(0, 'x', 'x')
        FY = FY / FY.sum(axis=(-1, -2)).dimshuffle(0, 'x', 'x')

        #W = f_batched_dot(FY, image)
        #return W

        W = f_batched_dot(f_batched_dot(FY, image), FX.transpose([0,2,1]))
        return W
        return W.reshape((batch_size, N*N))
        """

        
def read(I, N, g, delta, sigma):
    """Readout (scaled) windows of NxN pixels from the images I

    Parameters
    ----------
        I : numpy.ndarray (shape: (batch_size, height, width) )
            batch of input images
        N : int  
            $N \times N$ will be the size of the returned extracted 
            attention window
        g : numpy.ndarray of shape (batch_size, 2)
            $g_y, $g_x$ represent the center position where to place the
            attention window
        delta : numpy.ndarray of shape (batch_size,)
            distance between the attention window pixels
        width : numpy.ndarray of shape (batch_size,)
            
        
    Returns
    -------
        
    """
    batch_size, sY, sX = I.shape

    gx = g[:,1]
    gy = g[:,0]

    muX = gx[:,None] + delta[:,None]*(np.arange(N)[None,:] - N/2. - .5)
    muY = gy[:,None] + delta[:,None]*(np.arange(N)[None,:] - N/2. - .5)
    
    a = np.arange(sX)
    b = np.arange(sY)

    FX = np.exp( - (a[None, None,:] - muX[:,:,None])**2 / 2. / sigma[:,None,None]**2) 
    FY = np.exp( - (b[None, None,:] - muY[:,:,None])**2 / 2. / sigma[:,None,None]**2)

    #import ipdb; ipdb.set_trace()
    FX = FX / FX.sum(axis=(1, 2))[:, None, None]
    FY = FY / FY.sum(axis=(1, 2))[:, None, None]
    
    #return FX, FY, I, batched_dot(FY, I)
    return batched_dot(batched_dot(FY, I), FX.transpose([0,2,1]))

def write(C, W, g, delta, sigma):
    N, _ = W.shape
    sY, sX = C.shape

    muY = gy + delta*(np.arange(N) - N/2. - .5)
    muX = gx + delta*(np.arange(N) - N/2. - .5)
    
    a = np.arange(sX)
    b = np.arange(sY)

    FX = np.exp( - (a[None,:] - muX[:,None])**2 / 2. / sigma**2) 
    FY = np.exp( - (b[None,:] - muY[:,None])**2 / 2. / sigma**2) 
    FX /= FX.sum()
    FY /= FY.sum()
 
    C += np.dot(np.dot(FY.T, W), FX)
    return C

if __name__ == "__main__":
    from PIL import Image

    N = 40 
    height = 480
    width =  640

    #------------------------------------------------------------------------
    att = ZoomableAttentionWindow(height, width, N, normalize=True)

    I_ = T.matrix()
    center_y_ = T.vector()
    center_x_ = T.vector()
    delta_ = T.vector()
    sigma_ = T.vector()
    W_ = att.read(I_, center_y_, center_x_, delta_, sigma_)

    do_read = theano.function(inputs=[I_, center_y_, center_x_, delta_, sigma_],
                              outputs=W_, allow_input_downcast=True)

    W_ = T.matrix()
    center_y_ = T.vector()
    center_x_ = T.vector()
    delta_ = T.vector()
    sigma_ = T.vector()
    I_ = att.write(W_, center_y_, center_x_, delta_, sigma_)

    do_write = theano.function(inputs=[W_, center_y_, center_x_, delta_, sigma_],
                              outputs=I_, allow_input_downcast=True)

    #------------------------------------------------------------------------

    I = Image.open("cat.jpg")
    I = I.resize((640, 480)).convert('L')
    
    I = np.asarray(I).reshape( (width*height) )
    I = I / 255.

    center_y = 0.5 # 200
    center_x = 0.5 # 330
    delta = 0.3
    sigma = 2.

    I, center_y, center_x, delta, sigma = \
        vectorize(I, np.array(center_y), np.array(center_x), np.array(delta), np.array(sigma))

    #import ipdb; ipdb.set_trace()

    W  = do_read(I, center_y, center_x, delta, sigma)
    I2 = do_write(W, center_y, center_x, delta, sigma)
    
    import pylab
    pylab.figure()
    pylab.gray()
    pylab.imshow(I.reshape([height, width]), interpolation='nearest')

    pylab.figure()
    pylab.gray()
    pylab.imshow(W.reshape([N, N]), interpolation='nearest')

    pylab.figure()
    pylab.gray()
    pylab.imshow(I2.reshape([height, width]), interpolation='nearest')
    pylab.show(block=True)
    
