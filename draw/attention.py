#!/ysr/bin/env python 

from __future__ import division

import numpy as np

import theano 
import theano.tensor as T

from theano import tensor



class ZoomableAttentionWindow(object):
    def __init__(self, img_height, img_width, N):
        """A zoomable attention window for images.

        Parameters
        ----------
        img_heigt, img_width : int
            shape of the images 
        N : 
            $N \times N$ attention window size
        """
        self.img_height = img_height
        self.img_width = img_width
        self.N = N

    def filterbank_matrices(self, center_y, center_x, delta, sigma):
        """Create a Fy and a Fx
        
        Parameters
        ----------
        center_y : T.vector (shape: batch_size)
        center_x : T.vector (shape: batch_size)
            Y and X center coordinates for the attention window
        delta : T.vector (shape: batch_size)
        sigma : T.vector (shape: batch_size)
        
        Returns
        -------
            FY, FX 
        """
        tol = 1e-4
        N = self.N

        muX = center_x.dimshuffle([0, 'x']) + delta.dimshuffle([0, 'x'])*(T.arange(N)-N/2-0.5)
        muY = center_y.dimshuffle([0, 'x']) + delta.dimshuffle([0, 'x'])*(T.arange(N)-N/2-0.5)

        a = tensor.arange(self.img_width)
        b = tensor.arange(self.img_height)
        
        FX = tensor.exp( -(a-muX.dimshuffle([0,1,'x']))**2 / 2. / sigma.dimshuffle([0,'x','x'])**2 )
        FY = tensor.exp( -(b-muY.dimshuffle([0,1,'x']))**2 / 2. / sigma.dimshuffle([0,'x','x'])**2 )
        FX = FX / (FX.sum(axis=-1).dimshuffle(0, 1, 'x') + tol)
        FY = FY / (FY.sum(axis=-1).dimshuffle(0, 1, 'x') + tol)

        return FY, FX


    def read(self, images, center_y, center_x, delta, sigma):
        """
        Parameters
        ----------
        images : T.matrix    (shape: batch_size x img_size)
            Batch of images. Internally it will be reshaped to be a 
            (batch_size, img_height, img_width)-shaped stack of images.
        center_y : T.vector (shape: batch_size)
        center_x : T.vector (shape: batch_size)
        delta : T.vector    (shape: batch_size)
        sigma : T.vector    (shape: batch_size)

        Returns
        -------
        window : T.matrix   (shape: batch_size x N**2)
        """
        N = self.N
        batch_size = images.shape[0]

        # Reshape input into proper 2d images
        I = images.reshape( (batch_size, self.img_height, self.img_width) )

        # Get separable filterbank
        FY, FX = self.filterbank_matrices(center_y, center_x, delta, sigma)

        # apply to the batch of images
        W = T.batched_dot(T.batched_dot(FY, I), FX.transpose([0,2,1]))

        return W.reshape((batch_size, N*N))

    def write(self, windows, center_y, center_x, delta, sigma):
        N = self.N
        batch_size = windows.shape[0]

        # Reshape input into proper 2d windows
        W = windows.reshape( (batch_size, N, N) )

        # Get separable filterbank
        FY, FX = self.filterbank_matrices(center_y, center_x, delta, sigma)

        # apply...
        I = T.batched_dot(T.batched_dot(FY.transpose([0,2,1]), W), FX)

        return I.reshape( (batch_size, self.img_height*self.img_width) )

    def nn2att(self, l):
        """Convert neural-net outputs to attention parameters
    
        Parameters
        ----------
        l : tensor (batch_size x 5)
    
        Returns
        -------
        center_y : vector (batch_size)
        center_x : vector (batch_size)
        delta : vector (batch_size)
        sigma : vector (batch_size)
        gamma : vector (batch_size)
        """
        center_y  = l[:,0]
        center_x  = l[:,1]
        log_delta = l[:,2]
        log_sigma = l[:,3]
        log_gamma = l[:,4]
    
        delta = T.exp(log_delta)
        sigma = T.exp(log_sigma/2.)
        gamma = T.exp(log_gamma).dimshuffle(0, 'x')
    
        # normalize coordinates
        center_x = (center_x+1.)/2. * self.img_width
        center_y = (center_y+1.)/2. * self.img_height
        delta = (max(self.img_width, self.img_height)-1) / (self.N-1) * delta
    
        return center_y, center_x, delta, sigma, gamma


#=============================================================================


if __name__ == "__main__":
    from PIL import Image

    N = 40 
    height = 480
    width =  640

    #------------------------------------------------------------------------
    att = ZoomableAttentionWindow(height, width, N)

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

    center_y = 200.5
    center_x = 330.5
    delta = 5.
    sigma = 2.

    def vectorize(*args):
        return [a.reshape((1,)+a.shape) for a in args]

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
    
    import ipdb; ipdb.set_trace()
