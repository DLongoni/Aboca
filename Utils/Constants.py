#!/usr/bin/env python

from collections import OrderedDict

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
abc_l = list(ab_colors.values())

pic_path = './Fig/'
