import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from GA.ga_fitness import evaluate
from GA.ga_population import create_individual
from GA.ga_main import run_ga 
import pandas as pd
import Funciones_Superconductores as FSC
import matplotlib.pyplot as plt

x_p = pd.read_csv('data/X_data_clean.csv')
y_p = pd.read_csv('data/y_data_clean.csv')
seeds = [11, 21, 31, 41, 51]
N_gen = 100
Pop_size = 50
Mutpb = 0.5
df_1 = pd.DataFrame()


for i in range (len(seeds)):
    population, df_hist, df_top, df_resumen = run_ga(
        pop_size=Pop_size,
        ngen=N_gen,
        cxpb=0.5,
        mutpb=Mutpb,
        seed=seeds[i],
        run_id=i,
        verbose=True
    )
    df_2 = df_top.head(1)
    df_1 = pd.concat([df_1, df_2], ignore_index=True)
    print (f"Corrida {i+1} terminada")
df_mejores = df_1.sort_values("fitness", ascending=True).reset_index(drop=True)

X_dataset = x_p.values
X_candidates, df_mejores2 = FSC.composiciones_a_matriz_y_formula(df_mejores, x_p.columns)
distancias = FSC.calcular_knn(X_dataset, X_candidates)

df_mejores["knn_dist"] = distancias[:,0]

print(df_top)
#print(df_resumen)
#print(df_1)
#print(df_mejores)
print(df_mejores2)
#FSC.PCA_graf(x_p,X_candidates)
FSC.umap_dataset_y_candidatos(x_p, y_p, X_candidates)
plt.show()