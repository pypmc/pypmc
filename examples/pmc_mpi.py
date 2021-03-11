'''This example shows how to use importance sampling and how to
adapt the proposal density using the pmc algorithm in an MPI
parallel environment.
In order to have a multiprocessing enviroment invoke this script with
"mpirun -n 10 python pmc_mpi.py".

'''


from mpi4py.MPI import COMM_WORLD as comm

import numpy as np
import pypmc
import pypmc.tools.parallel_sampler # this submodule is NOT imported by ``import pypmc``

# This script is a parallelized version of the PMC example ``pmc.py``.
# The following lines just define a target density and an initial proposal.
# These steps are exactly the same as in ``pmc.py``:

# define the target; i.e., the function you want to sample from.
# In this case, it is a bimodal Gaussian
#
# Note that the target function "log_target" returns the log of the
# target function.
component_weights = np.array([0.3, 0.7])

mean0       = np.array ([ 5.0  , 0.01  ])
covariance0 = np.array([[ 0.01 , 0.003 ],
                        [ 0.003, 0.0025]])
inv_covariance0 = np.linalg.inv(covariance0)

mean1       = np.array ([-4.0  , 1.0   ])
covariance1 = np.array([[ 0.1  , 0.    ],
                        [ 0.   , 0.02  ]])
inv_covariance1 = np.linalg.inv(covariance1)

component_means = [mean0, mean1]
component_covariances = [covariance0, covariance1]

target_mixture = pypmc.density.mixture.create_gaussian_mixture(component_means, component_covariances, component_weights)

log_target = target_mixture.evaluate


# define the initial proposal density
# In this case it has three Gaussians:
# the initial covariances are set to the unit-matrix,
# the initial component weights are set equal
initial_prop_means = []
initial_prop_means.append( np.array([ 4.0, 0.0]) )
initial_prop_means.append( np.array([-5.0, 0.0]) )
initial_prop_means.append( np.array([ 0.0, 0.0]) )
initial_prop_covariance = np.eye(2)

initial_prop_components = []
for i in range(3):
    initial_prop_components.append(pypmc.density.gauss.Gauss(initial_prop_means[i], initial_prop_covariance))

initial_proposal = pypmc.density.mixture.MixtureDensity(initial_prop_components)

# -----------------------------------------------------------------------------------------------------------------------

# In ``pmc.py`` the following line defines the sequential single process sampler:
# sampler = pypmc.sampler.importance_sampling.ImportanceSampler(log_target, initial_proposal)
#
# We now use the parallel MPISampler instead:
SequentialIS = pypmc.sampler.importance_sampling.ImportanceSampler
parallel_sampler = pypmc.tools.parallel_sampler.MPISampler(SequentialIS, target=log_target, proposal=initial_proposal)

# Draw 10,000 samples adapting the proposal every 1,000 samples:

# make sure that every process has a different random number generator seed
if comm.Get_rank() == 0:
    seed = np.random.randint(1e5)
else:
    seed = None
seed = comm.bcast(seed)
np.random.seed(seed + comm.Get_rank())

generating_components = []
for i in range(10):
    # With the invocation "mpirun -n 10 python pmc_mpi.py", there are
    # 10 processes which means in order to draw 1,000 samples
    # ``parallel_sampler.run(1000//comm.Get_size())`` makes each process draw
    # 100 samples.
    # Hereby the generating proposal component for each sample in each process
    # is returned by ``parallel_sampler.run``.
    # In the master process, ``parallel_sampler.run`` is a list containing the
    # return values of the sequential ``run`` method of every process.
    # In all other processes, ``parallel_sampler.run`` returns the generating
    # component for its own samples only.
    last_generating_components = parallel_sampler.run(1000//comm.Get_size(), trace_sort=True)

    # In addition to the generating components, the ``sampler.run``
    # method automatically sends all samples to the master
    # process i.e. the process which fulfills comm.Get_rank() == 0.
    if comm.Get_rank() == 0:
        print("\rstep", i, "...\n\t", end='')

        # Now let PMC run only in the master process:

        # ``sampler.samples_list`` and ``sampler.weights_list`` store the weighted samples
        # sorted by the resposible process:
        # The History objects that are held by process i can be accessed via
        # ``sampler.<samples/weights>_list[i]``. The master process (i=0) also produces samples.

        # Combine the weights and samples to two arrays of 1,000 samples
        samples = np.vstack([history_item[-1] for history_item in parallel_sampler.samples_list])
        weights = np.vstack([history_item[-1] for history_item in parallel_sampler.weights_list])[:,0]

        # The latent variables are stored in ``last_generating_components``.
        # ``last_generating_components[i]`` returns an array with the generating
        # components of the samples produced by process number "i".
        # ``np.hstack(last_generating_components)`` combines the generating components
        # from all processes to one array holding all 1,000 entries.
        generating_components.append(  np.hstack(last_generating_components)  )

        # adapt the proposal using the samples from all processes
        new_proposal =  pypmc.mix_adapt.pmc.gaussian_pmc(samples, parallel_sampler.sampler.proposal,
                                                         weights, generating_components[-1],
                                                         mincount=20, rb=True)
    else:
        # In order to broadcast the ``new_proposal``, define a dummy variable in the other processes
        # see "MPI4Py tutorial", section "Collective Communication": http://mpi4py.scipy.org/docs/usrman/tutorial.html
        new_proposal = None

    # broadcast the ``new_proposal``
    new_proposal = comm.bcast(new_proposal)

    # replace the old proposal
    parallel_sampler.sampler.proposal = new_proposal

# only the master process shall print out any final information
if comm.Get_rank() == 0:
    all_samples  = np.vstack([history_item[ :] for history_item in parallel_sampler.samples_list])
    all_weights  = np.vstack([history_item[ :] for history_item in parallel_sampler.weights_list])
    last_samples = np.vstack([history_item[-1] for history_item in parallel_sampler.samples_list])
    last_weights = np.vstack([history_item[-1] for history_item in parallel_sampler.weights_list])
    print("\rsampling finished", end=', ')
    print("collected " + str(len(all_samples)) + " samples")
    print(  '------------------------------------------')
    print('\n')

    # print information about the adapted proposal
    print('initial component weights:', initial_proposal.weights)
    print('final   component weights:', parallel_sampler.sampler.proposal.weights)
    print('target  component weights:', component_weights)
    print()
    for k, m in enumerate([mean0, mean1, None]):
        print('initial mean of component %i:' %k, initial_proposal.components[k].mu)
        print('final   mean of component %i:' %k, parallel_sampler.sampler.proposal.components[k].mu)
        print('target  mean of component %i:' %k, m)
        print()
    print()
    for k, c in enumerate([covariance0, covariance1, None]):
        print('initial covariance of component %i:\n' %k, initial_proposal.components[k].sigma, sep='')
        print()
        print('final   covariance of component %i:\n' %k, parallel_sampler.sampler.proposal.components[k].sigma, sep='')
        print()
        print('target  covariance of component %i:\n' %k, c, sep='')
        print('\n')

    if comm.Get_size() == 1:
        print('******************************************************')
        print('********** NOTE: There is only one process. **********')
        print('******** try "mpirun -n 10 python pmc_mpi.py" ********')
        print('******************************************************')

    # plot results
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('For plotting "matplotlib" needs to be installed')
        exit(1)

    def set_axlimits():
        plt.xlim(-6.0, +6.000)
        plt.ylim(-0.2, +1.401)

    plt.subplot(221)
    plt.title('target mixture')
    pypmc.tools.plot_mixture(target_mixture, cmap='jet')
    set_axlimits()

    plt.subplot(222)
    plt.title('pmc fit')
    pypmc.tools.plot_mixture(parallel_sampler.sampler.proposal, cmap='nipy_spectral', cutoff=0.01)
    set_axlimits()

    plt.subplot(223)
    plt.title('target mixture and pmc fit')
    pypmc.tools.plot_mixture(target_mixture, cmap='jet')
    pypmc.tools.plot_mixture(parallel_sampler.sampler.proposal, cmap='nipy_spectral', cutoff=0.01)
    set_axlimits()

    plt.subplot(224)
    plt.title('weighted samples')
    plt.hist2d(last_samples[:,0], last_samples[:,1], weights=last_weights[:,0], cmap='gray_r', bins=200)
    set_axlimits()

    plt.tight_layout()
    plt.show()
