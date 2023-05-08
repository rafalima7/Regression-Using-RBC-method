# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 15:57:06 2023

@author: Rafael Lima
"""

import pandas as pd
import seaborn as sns


campos1 = pd.read_csv('dado_campos1.csv', sep='\s+')

campos1

sns.pairplot(campos1)
