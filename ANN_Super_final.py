# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 14:11:14 2025

@author: giorn
"""

# ====== Imports básicos ======
import numpy as np                 # cálculo numérico
import pandas as pd               # manejo de datos/tablas
import torch                      # tensor computation (PyTorch)
import torch.nn as nn             # capas, pérdidas.
import torch.optim as optim       # optimizadores (AdamW)
import torch.nn.functional as F   # funciones (ReLU, softplus, dropout)

from sklearn.model_selection import train_test_split     # split train/val/test
from sklearn.preprocessing import StandardScaler         # estandarización (z-score)
from sklearn.metrics import r2_score                     # R^2 (métrica de regresión)
import Funciones_Superconductores as FSC

import matplotlib.pyplot as plt    # gráficos

#   1       ----------------------- Datos------------------------

# ===========================================================
CSV_PATH = "Data_base_supercond.csv"

# FUNCIÓN 1: ABS + NORMALIZACIÓN
# ===========================================================

Xn,yn,df = FSC.Normalizar(CSV_PATH)

# FUNCIÓN 2: ELIMINAR COMPOSICIONES REPETIDAS 
# ===========================================================
X,y = FSC.Eliminar_Comp_repetidas(Xn,yn)

# SPLIT DE LOS DATOS 
#============================================================
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42)



# -------- Tensores ---------------------------------------------
X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
X_test_t  = torch.tensor(X_test.values,  dtype=torch.float32)

y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test_t  = torch.tensor(y_test.values,  dtype=torch.float32).unsqueeze(1) 


# 4 -------------- Modelo -------------------------------------

class Semiconductor_Tc(nn.Module):
    def __init__(self, d_ini):
        super().__init__()
        self.fc1 = nn.Linear(d_ini, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 50)
        self.bn2 = nn.BatchNorm1d(50)
        self.fc3 = nn.Linear(50, 20)
        self.out = nn.Linear(20, 1)
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.fc3(x))
    
        z = self.out(x)
        z = F.softplus(z)
        return z
    
N_ini = X_train_t.shape[1]
modelo = Semiconductor_Tc(N_ini)

# 
# 5) ENTRENAMIENTO 
# ===========================================================
loss_fun = nn.SmoothL1Loss()
opt = optim.AdamW(modelo.parameters(), lr=1e-4, weight_decay=1e-4)

max_epochs = 100
batch_size = 50

for epoch in range(max_epochs):
    modelo.train()
    idx_perm = torch.randperm(len(X_train_t))

    for i in range(0, len(X_train_t), batch_size):
        b = idx_perm[i:i+batch_size]
        y_pred = modelo(X_train_t[b])
        loss = loss_fun(y_pred, y_train_t[b])

        opt.zero_grad()
        loss.backward()
        opt.step()

    # log cada 20 épocas (solo loss de train)
    if (epoch + 1) % 20 == 0 or epoch == 0:
        modelo.eval()
        with torch.no_grad():
            y_tr_pred = modelo(X_train_t)
            tr_loss = loss_fun(y_tr_pred, y_train_t).item()
        print(f"Época {epoch+1:3d} | Train loss: {tr_loss:.4f}")

modelo.eval()

with torch.no_grad():
    z_train = modelo(X_train_t)  # (N,1)
    z_test  = modelo(X_test_t)

# a numpy 1D (para FSC.mostrar_resultados o tu función)
y_pred_train = z_train.cpu().numpy().ravel()
y_pred_test  = z_test.cpu().numpy().ravel()

y_train_np = np.array(y_train).ravel()
y_test_np  = np.array(y_test).ravel()

r2_train = r2_score(y_train_np, y_pred_train)
r2_test  = r2_score(y_test_np,  y_pred_test)

print(f"\nR² (train): {r2_train:.4f}")
print(f"R² (test) : {r2_test:.4f}")
# ---- A partir de aquí tú llamas tus funciones de gráficas, por ejemplo:
FSC.mostrar_resultados(y_train_np, y_pred_train, "train")
FSC.mostrar_resultados(y_test_np,  y_pred_test,  "test")




