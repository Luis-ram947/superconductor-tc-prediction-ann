import random
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
from joblib import load
import Funciones_Superconductores as FSC
import matplotlib.pyplot as plt




# Datos crudos 
CSV_PATH = pd.read_csv('data/Data_base_supercond.csv')

x = CSV_PATH.iloc[:,:-1]   # Composiciones quimicas crudas
y = CSV_PATH.iloc[:,-1]    # Temperatura critica de super conductividad 

# Datos procesados
x_p = pd.read_csv('data/X_data_clean.csv')
y_p = pd.read_csv('data/y_data_clean.csv').iloc[:, 0]



# RF entrenado
RF_model  = load("models/rf_model.pkl")

y_predic = RF_model.predict(x_p)

FSC.histograma(y,40,0.5, "Datos crudos")
FSC.histograma(y_p,40,0.5, "Datos procesados")
FSC.residuos(y_p,y_predic,"Predicciones")
FSC.Paridad(y_p,y_predic,"Predicciones")
FSC.comportamiento_con_error(y_p,y_predic)
FSC.grafica_y_ordenada(y)
FSC.grafica_y_ordenada(y_p)
matriz_freq = FSC.heatmap_elementos_tc(x_p, y_p, top_n=30)
FSC.superficie_3D(matriz_freq)
FSC.frecuencia_elementos(x_p)
plt.show()
