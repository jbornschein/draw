import logging

from blocks.algorithms import GradientDescent, RMSProp
from blocks.bricks import MLP, Linear, Rectifier
from blocks.bricks.parallel import Fork
from blocks.datasets.mnist import MNIST
from blocks.datasets.streams import DataStream
from blocks.datasets.schemes import SequentialScheme
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, IsotropicGaussian
from blocks.main_loop import MainLoop
from blocks.roles import add_role, PARAMETER
from blocks.utils import shared_floatx
import numpy
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams


def main():
    nvis, nhid, nlat, learn_prior = 784, 200, 100, False
    theano_rng = MRG_RandomStreams(134663)

    # Initialize prior
    prior_mu = shared_floatx(numpy.zeros(nlat), name='prior_mu')
    prior_log_sigma = shared_floatx(numpy.zeros(nlat), name='prior_log_sigma')
    if learn_prior:
        add_role(prior_mu, PARAMETER)
        add_role(prior_log_sigma, PARAMETER)

    # Initialize encoding network
    encoding_network = MLP(activations=[Rectifier()],
                           dims=[nvis, nhid],
                           weights_init=IsotropicGaussian(std=0.001),
                           biases_init=Constant(0))
    encoding_network.initialize()
    encoding_parameter_mapping = Fork(
        output_names=['mu_phi', 'log_sigma_phi'], input_dim=nhid,
        output_dims=dict(mu_phi=nlat, log_sigma_phi=nlat), prototype=Linear(),
        weights_init=IsotropicGaussian(std=0.001), biases_init=Constant(0))
    encoding_parameter_mapping.initialize()

    # Initialize decoding network
    decoding_network = MLP(activations=[Rectifier()],
                           dims=[nlat, nhid],
                           weights_init=IsotropicGaussian(std=0.001),
                           biases_init=Constant(0))
    decoding_network.initialize()
    decoding_parameter_mapping = Linear(
        input_dim=nhid, output_dim=nvis, name='mu_theta',
        weights_init=IsotropicGaussian(std=0.001),
        biases_init=Constant(0))
    decoding_parameter_mapping.initialize()

    # Encode / decode
    x = tensor.matrix('features')
    h_phi = encoding_network.apply(x)
    mu_phi, log_sigma_phi = encoding_parameter_mapping.apply(h_phi)
    epsilon = theano_rng.normal(size=mu_phi.shape, dtype=mu_phi.dtype)
    epsilon.name = 'epsilon'
    z = mu_phi + epsilon * tensor.exp(log_sigma_phi)
    z.name = 'z'
    h_theta = decoding_network.apply(z)
    mu_theta = decoding_parameter_mapping.apply(h_theta)

    # Compute cost
    kl_term = (
        prior_log_sigma - log_sigma_phi
        + 0.5 * (
            tensor.exp(2 * log_sigma_phi) + (mu_phi - prior_mu) ** 2
        ) / tensor.exp(2 * prior_log_sigma)
        - 0.5
    ).sum(axis=1)
    kl_term.name = 'kl_term'
    kl_term_mean = kl_term.mean()
    kl_term_mean.name = 'avg_kl_term'
    reconstruction_term = - (
        x * tensor.nnet.softplus(-mu_theta)
        + (1 - x) * tensor.nnet.softplus(mu_theta)).sum(axis=1)
    reconstruction_term.name = 'reconstruction_term'
    reconstruction_term_mean = -reconstruction_term.mean()
    reconstruction_term_mean.name = 'avg_reconstruction_term'
    cost = -(reconstruction_term - kl_term).mean()
    cost.name = 'nll_upper_bound'

    # Datasets and data streams
    mnist_train = MNIST(
        'train', start=0, stop=50000, binary=True, sources=('features',))
    train_loop_stream = DataStream(
        dataset=mnist_train,
        iteration_scheme=SequentialScheme(mnist_train.num_examples, 100))
    train_monitor_stream = DataStream(
        dataset=mnist_train,
        iteration_scheme=SequentialScheme(mnist_train.num_examples, 500))
    mnist_valid = MNIST(
        'train', start=50000, stop=60000, binary=True, sources=('features',))
    valid_monitor_stream = DataStream(
        dataset=mnist_valid,
        iteration_scheme=SequentialScheme(mnist_valid.num_examples, 500))
    mnist_test = MNIST('test', binary=True, sources=('features',))
    test_monitor_stream = DataStream(
        dataset=mnist_test,
        iteration_scheme=SequentialScheme(mnist_test.num_examples, 500))

    # Get parameters
    computation_graph = ComputationGraph([cost])
    params = VariableFilter(roles=[PARAMETER])(computation_graph.variables)

    # Training loop
    step_rule = RMSProp(learning_rate=1e-3, decay_rate=0.95)
    algorithm = GradientDescent(cost=cost, params=params, step_rule=step_rule)
    monitored_quantities = [cost, reconstruction_term_mean, kl_term_mean]
    main_loop = MainLoop(
        model=None, data_stream=train_loop_stream, algorithm=algorithm,
        extensions=[
            Timing(),
            FinishAfter(after_n_epochs=200),
            DataStreamMonitoring(
                monitored_quantities, train_monitor_stream, prefix="train"),
            DataStreamMonitoring(
                monitored_quantities, valid_monitor_stream, prefix="valid"),
            DataStreamMonitoring(
                monitored_quantities, test_monitor_stream, prefix="test"),
            Printing()])
    main_loop.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
