'''Functions to evaluate the pdf of specific probability distributions

'''

from math import exp, log
import numpy as np

def unnormalized_log_pdf_gauss(x, mu, inv_sigma):
    return - .5 * (x-mu).dot(inv_sigma).dot(x-mu)

def normalized_pdf_gauss(x, mu, inv_sigma):
    return exp( unnormalized_log_pdf_gauss(x, mu, inv_sigma) - .5 * len(mu) * log(2.*np.pi) + .5 * log(np.linalg.det(inv_sigma)) )
