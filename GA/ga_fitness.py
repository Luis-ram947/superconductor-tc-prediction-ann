import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
#from deap import base, creator, tools
from Chemical_domain import chemical_domain as CD
from joblib import load


RF_model  = load("models/rf_model.pkl")


def evaluate (individual,Tc_target=100):
    x = list(individual)
    x = CD.repair(x)
    if not CD.is_feasible_composition(x):
        return (1e6,)
    x_df = pd.DataFrame([x], columns=CD.ELEMENTS)
    Tc = RF_model.predict(x_df)[0]


    error = abs(Tc-Tc_target)
    return (error,)
    