#-*- coding:utf-8 -*-
""" pac_bound_zero(...) function.
This file can be imported in your python project or executed as a command-line script.

See the related paper:
Risk Bounds for the Majority Vote: From a PAC-Bayesian Analysis to a Learning Algorithm
by Germain, Lacasse, Laviolette, Marchand and Roy (JMLR 2015)

http://graal.ift.ulaval.ca/majorityvote/
"""
__author__ = 'Pascal Germain'

from pac_bound_tools import validate_inputs, xi, solve_kl_sup
from math import log

def pac_bound_zero(empirical_gibbs_risk, m, KLQP, delta=0.05):
    """ PAC Bound ZERO of Germain, Lacasse, Laviolette, Marchand and Roy (JMLR 2015)

    Compute a PAC-Bayesian upper bound on the Bayes risk by
    multiplying by two an upper bound on the Gibbs risk

    empirical_gibbs_risk : Gibbs risk on the training set
    m : number of training examples
    KLQP : Kullback-Leibler divergence between prior and posterior
    delta : confidence parameter (default=0.05)
    """
    if not validate_inputs(empirical_gibbs_risk, None, m, KLQP, delta): return 1.0

    xi_m = xi(m)
    right_hand_side = ( KLQP + log( xi_m / delta ) ) / m
    sup_R = min(0.5, solve_kl_sup(empirical_gibbs_risk, right_hand_side))

    return 2 * sup_R


if __name__ == '__main__':
    from sys import argv
    from collections import OrderedDict

    argc = len(argv)
    if argc < 2 :
        print('-'*100)
        print('Usage: pac_bound_0.py empirical_gibbs_risk [m] [KLQP] [delta]')
        print('-'*100)
        print(pac_bound_zero.func_doc)
    else:
        arg_dict = OrderedDict()
        arg_dict['empirical_gibbs_risk']    = float(argv[1])
        arg_dict['m']                       = int(argv[2])   if argc > 2 else 1000
        arg_dict['KLQP']                    = float(argv[3]) if argc > 3 else 5.0
        arg_dict['delta']                   = float(argv[4]) if argc > 4 else 0.05

        for key, value in arg_dict.iteritems(): print(key + ' = ' + str(value))

        bound = pac_bound_zero(**arg_dict)
        print('bayes risk bound = %f' % bound)
