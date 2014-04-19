#-*- coding:utf-8 -*-
""" pac_bound_one(...) function.
This file can be imported in your python project or executed as a command-line script.

See the related paper:
Risk Bounds for the Majority Vote: From a PAC-Bayesian Analysis to a Learning Algorithm
by Germain, Lacasse, Laviolette, Marchand and Roy (JMLR, 2014)

http://graal.ift.ulaval.ca/majorityvote/
"""
__author__ = 'Pascal Germain'

from pac_bound_tools import validate_inputs, xi, solve_kl_inf, solve_kl_sup, c_bound_third_form
from math import log

def pac_bound_one(empirical_gibbs_risk, empirical_disagreement, m, KLQP, delta=0.05):
    """ PAC Bound ONE of Germain, Lacasse, Laviolette, Marchand and Roy (JMLR, 2014)

    Compute a PAC-Bayesian upper bound on the Bayes risk by
    using the C-Bound on an upper bound on the Gibbs risk
    and a lower bound on the expected disagreement

    empirical_gibbs_risk : Gibbs risk on the training set
    empirical_disagreement : Expected disagreement on the training set
    m : number of training examples
    KLQP : Kullback-Leibler divergence between prior and posterior
    delta : confidence parameter (default=0.05)
    """
    if not validate_inputs(empirical_gibbs_risk, empirical_disagreement, m, KLQP, delta): return 1.0

    xi_m = xi(m)
    right_hand_side = ( KLQP + log( 2 * xi_m / delta ) ) / m
    sup_R = min(0.5, solve_kl_sup(empirical_gibbs_risk, right_hand_side))

    right_hand_side = ( 2*KLQP + log( 2 * xi_m / delta ) ) / m
    inf_D = solve_kl_inf(empirical_disagreement, right_hand_side)

    return c_bound_third_form(sup_R, inf_D)


if __name__ == '__main__':
    from sys import argv
    from collections import OrderedDict

    argc = len(argv)
    if argc < 3 :
        print('-'*100)
        print('Usage: pac_bound_1.py empirical_gibbs_risk empirical_disagreement [m] [KLQP] [delta]')
        print('-'*100)
        print(pac_bound_one.func_doc)
    else:
        arg_dict = OrderedDict()
        arg_dict['empirical_gibbs_risk']    = float(argv[1])
        arg_dict['empirical_disagreement']  = float(argv[2])
        arg_dict['m']                       = int(argv[3])   if argc > 3 else 1000
        arg_dict['KLQP']                    = float(argv[4]) if argc > 4 else 5.0
        arg_dict['delta']                   = float(argv[5]) if argc > 5 else 0.05

        for key, value in arg_dict.iteritems(): print(key + ' = ' + str(value))

        bound = pac_bound_one(**arg_dict)
        print('bayes risk bound = %f' % bound)
