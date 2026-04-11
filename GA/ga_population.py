import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
#from deap import base, creator, tools
from Chemical_domain import chemical_domain as CD

def create_individual():
    return CD.generate_valid_composition()