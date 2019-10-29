#!/usr/bin/env python

from collections import OrderedDict
import numpy as np

colors = ['b', 'g', 'r', 'c', 'y', 'm', 'k', '#999999']
ab_colors = OrderedDict([
    ('azzurro', '#B9D8EC'),
    ('verde', '#88C86E'),
    ('tortora', '#AFAC7C'),
    ('giallo', '#CAD401'),
    ('rosso', '#7C232B'),
    ('nero', 'k'),
    ('grigio', '#9D9B8D')
])

abc = ab_colors
abc_l = np.array(list(ab_colors.values()))
abcl_l = abc_l[[1, 3, 4, 2, 0, 6, 5]]

pic_path = './Fig/'

BASE_DB = "it"


def dumps_path(db=BASE_DB):
    ret = "./Dataset/" + db + "/"
    return ret
