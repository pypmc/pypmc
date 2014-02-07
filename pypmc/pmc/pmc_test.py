'''Unit tests for the Population Monte Carlo.

'''

from .pmc import *
from . import proposal, importance_sampling
from ..tools._probability_densities import normalized_pdf_gauss
from math import log
import numpy as np
import unittest

rng_seed  = 295627184
rng_steps = 20

# ------------------- define the weighted samples' context  -------------------

mu1 = np.array( [ 10., -1.0, 8.0] )
mu2 = np.array( [-10.,  7.4, 0.5] )

cov1 = np.array([[1.15 , 0.875, 0.0],
                 [0.875, 0.75 ,-0.2],
                 [0.0  ,-0.2  , 1.1]])

cov2 = np.array([[1.0  , 0.01 , 0.1],
                 [0.01 , 0.75 , 0.0],
                 [0.1  , 0.0  , 2.1]])

inv_cov1 = np.linalg.inv(cov1)
inv_cov2 = np.linalg.inv(cov2)

# target density
target_abundancies = np.array( (.6, .4) )
log_target = lambda x:   log(target_abundancies[0] * (normalized_pdf_gauss(x, mu1, inv_cov1)) + \
                             target_abundancies[1] * (normalized_pdf_gauss(x, mu2, inv_cov2)) ) \
                       + log(15.) # break normalization

# proposal density
gauss1 = proposal.GaussianComponent(mu1+.001, cov1+.001)
gauss2 = proposal.GaussianComponent(mu2-.005, cov2-.005)
proposal_abundancies = np.array( (.7, .3) )
prop = proposal.MixtureProposal((gauss1,gauss2), proposal_abundancies)

# -----------------------------------------------------------------------------

origins = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
weights = np.array([12.89295915,  12.89372694,  12.89781423,  12.79548829,
                    12.89397248,  12.88642498,  12.89875608,  12.8977244 ,
                    12.8834032 ,  12.81344527,  12.8966767 ,  12.89319812,
                    20.02787201,  19.89550322,  19.81661548,  19.9733172 ,
                    19.81867511,  19.81555008,  19.83955669,  19.83352245])
samples = np.array([[  9.7070033 ,  -1.14093259,   7.79492513],
                    [  9.56875908,  -1.3205348 ,   7.3705522 ],
                    [ 10.53728461,  -0.93171182,   8.76279014],
                    [  9.80289836,  -1.15107748,   9.27682257],
                    [  8.91717444,  -1.62000575,   7.60676764],
                    [  9.55705421,  -1.65785994,   9.4330834 ],
                    [ 10.90155376,  -0.42097835,   7.64481752],
                    [ 11.06838483,  -0.65188323,   8.69936008],
                    [  8.50673184,  -2.45559049,   8.62152455],
                    [ 10.8097935 ,  -0.33471831,   8.60497435],
                    [ 10.46129646,  -1.04132199,   9.04460811],
                    [ 10.23040728,  -0.63621386,   6.48880065],
                    [-10.76972316,   8.23669361,   2.06283074],
                    [-11.26019812,   7.03488615,  -0.87321151],
                    [ -9.99070915,   6.83422119,   0.28846651],
                    [ -9.39271812,   7.08741571,   1.91672609],
                    [-10.98814859,   7.55372701,   0.48618477],
                    [ -9.60983136,   6.24723833,   1.06241101],
                    [-10.61752466,   7.39052825,   1.17726011],
                    [-10.4898097 ,   7.48668861,  -2.41443733]])
weighted_samples = np.hstack( (weights.reshape(len(weights),1),samples) )

## ``origins``, ``weights`` and ``samples`` have been obtained using the following code:
#np.random.mtrand.seed(rng_seed)
#sampler = importance_sampling.ImportanceSampler(log_target, prop, rng=np.random.mtrand)
#new_origins = sampler.run(rng_steps, trace=True)
#new_weighted_samples = sampler.history[-1]

class TestPMC(unittest.TestCase):
    def test_gaussian_pmc_with_origin(self):
        #TODO: compare with pmclib (Kilbinger et. al.)
        adapted_prop = gaussian_pmc(weighted_samples, prop, origins)

        adapted_comp_weights = adapted_prop.weights
        adapted_mu1          = adapted_prop.components[0].mu
        adapted_mu2          = adapted_prop.components[1].mu
        adapted_sigma1       = adapted_prop.components[0].sigma
        adapted_sigma2       = adapted_prop.components[1].sigma

        # the values that should be obtained by the pmc algorithm
        pmc_comp_weights = np.array( (154.54358983999998, 159.02061223999999) ) / 313.56420207999997
        pmc_mu1          = np.array([ 1546.302278  ,  -172.1300429 ,  1279.34733595]) / 154.54358983999998
        pmc_mu2          = np.array([-1652.19922509,  1150.52591727,    74.098254  ]) / 159.02061223999999
        pmc_sigma1       = np.array([[91.13245238,  62.95055712,   4.96175291],
                                     [62.95055712,  51.04895641, -16.59026473],
                                     [ 4.96175291, -16.59026473, 111.63047879]]) / 154.54358983999998
        pmc_sigma2       = np.array([[ 61.35426434, -30.51320283,  50.0064872 ],
                                     [-30.51320283,  47.59366671,  10.77061072],
                                     [ 50.0064872 ,  10.77061072, 311.06169561]]) / 159.02061223999999

        np.testing.assert_allclose(adapted_comp_weights, pmc_comp_weights)
        np.testing.assert_allclose(adapted_mu1         , pmc_mu1         )
        np.testing.assert_allclose(adapted_mu2         , pmc_mu2         )
        np.testing.assert_allclose(adapted_sigma1      , pmc_sigma1      )
        np.testing.assert_allclose(adapted_sigma2      , pmc_sigma2      )

    def test_gaussian_pmc_without_origin(self):
        #TODO: compare with pmclib (Kilbinger et. al.)
        adapted_prop = gaussian_pmc(weighted_samples, prop)

        adapted_comp_weights = adapted_prop.weights
        adapted_mu1          = adapted_prop.components[0].mu
        adapted_mu2          = adapted_prop.components[1].mu
        adapted_sigma1       = adapted_prop.components[0].sigma
        adapted_sigma2       = adapted_prop.components[1].sigma

        # the values that should be obtained by the pmc algorithm
        pmc_comp_weights = np.array( (154.54358983999998, 159.02061223999999) ) / 313.56420207999997
        pmc_mu1          = np.array([1546.302278, -172.1300429, 1279.34733595]) / 154.54358983999998
        pmc_mu2          = np.array([-1652.19922509, 1150.52591727, 74.098254]) / 159.02061223999999
        pmc_sigma1       = np.array([[91.13245238,  62.95055712,   4.96175291],
                                     [62.95055712,  51.04895641, -16.59026473],
                                     [ 4.96175291, -16.59026473, 111.63047879]]) / 154.54358983999998
        pmc_sigma2       = np.array([[ 61.35426434, -30.51320283,  50.0064872 ],
                                     [-30.51320283,  47.59366671,  10.77061072],
                                     [ 50.0064872 ,  10.77061072, 311.06169561]]) / 159.02061223999999

        np.testing.assert_allclose(adapted_comp_weights, pmc_comp_weights)
        np.testing.assert_allclose(adapted_mu1         , pmc_mu1         )
        np.testing.assert_allclose(adapted_mu2         , pmc_mu2         )
        np.testing.assert_allclose(adapted_sigma1      , pmc_sigma1      )
        np.testing.assert_allclose(adapted_sigma2      , pmc_sigma2      )
