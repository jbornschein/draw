
import unittest

from blocks.bricks.recurrent import SimpleRecurrent, LSTM
from blocks.initialization import Constant, IsotropicGaussian, Orthogonal 

from draw.draw import *


#
class TestDRAW(unittest.TestCase):
    def setUp(self):
        n_iter = 2

        x_dim = 8
        z_dim = 10
        dec_dim = 12
        enc_dim = 16

        read_dim = 2*x_dim

        rnninits = {
            #'weights_init': Orthogonal(),
            'weights_init': IsotropicGaussian(0.01),
            'biases_init': Constant(0.),
        }
        inits = {
            #'weights_init': Orthogonal(),
            'weights_init': IsotropicGaussian(0.01),
            'biases_init': Constant(0.),
        }
 
        reader = Reader(x_dim=x_dim, dec_dim=dec_dim, **inits)
        writer = Writer(input_dim=dec_dim, output_dim=x_dim, **inits)

        encoder_rnn = LSTM(dim=enc_dim, name="RNN_enc", **rnninits)
        decoder_rnn = LSTM(dim=dec_dim, name="RNN_dec", **rnninits)
        encoder_mlp = MLP([Identity()], [(read_dim+dec_dim), 4*enc_dim], name="MLP_enc", **inits)
        decoder_mlp = MLP([Identity()], [             z_dim, 4*dec_dim], name="MLP_dec", **inits)
        q_sampler = Qsampler(input_dim=enc_dim, output_dim=z_dim, **inits)

        self.draw = DrawModel(n_iter, reader,
                        encoder_mlp, encoder_rnn, q_sampler,
                        decoder_mlp, decoder_rnn, writer)
        self.draw.initialize()

    def test_1(self):
        x = tensor.matrix('x')
        x_recons, kl_terms = self.draw.reconstruct(x)
        
        do_recons = theano.function(
                inputs=[x],
                outputs=[x_recons, kl_terms], 
                name="do_recons")

