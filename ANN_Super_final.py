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

import matplotlib.pyplot as plt    # gráficos

#   1       ----------------------- Datos------------------------

df = pd.read_csv(r"C:\Users\giorn\Desktop\pythoniza\Spyder\Pytorch\Superconduc\Data_base_supercond_clean.csv")

Columna_filtro = df.shape[1]-1   # Columna de y para filtrar datos muy pequeños 

Tmin = 0
Tmax = 250

Filas_ini = len(df)

# ---------- se extraen los indices en donde T se encuentra entre Tmin y Tmax ------
Indices = df.iloc[:, Columna_filtro].between(Tmin, Tmax, inclusive="both")  
df = df[Indices].reset_index(drop=True)

Filas_fin = len(df)
Filas_eli = Filas_ini - Filas_fin

X = df.iloc[:, 0:Columna_filtro]       # Entradas (Proporciones atómicas de elementos)
y = df.iloc[:, Columna_filtro]         # Salidas  (Temperatura crítica de supercond K)

# Nombre_columnas = df.columns[:86]   # guarda los nombres de los elementos si lo necesitas

N_tot = len(X)                      # Entradas totales

#  2  ---------------------- Separar datos test, validación y entrenamiento --------------

Indice_tot = np.arange(N_tot)    # arreglo de índices 

# ---------- 70% entrenamiento ------- 30% temporales ----------------------
Ind_train, Ind_tem = train_test_split(Indice_tot, test_size=0.3, shuffle=True)

# ---------------- de los 30% temp ------ 50% test, 50% val ---------
Ind_val, Ind_test = train_test_split(Ind_tem, test_size=0.5, shuffle=True)

# 3   -------------- Escalado de datos o normalizar datos --------

# Datos de entrenamiento 
X_train = X.iloc[Ind_train, :].to_numpy(); y_train = y.iloc[Ind_train].to_numpy()

# Datos de test 
X_test  = X.iloc[Ind_test,  :].to_numpy(); y_test  = y.iloc[Ind_test].to_numpy()

# Datos de val
X_val   = X.iloc[Ind_val,   :].to_numpy(); y_val   = y.iloc[Ind_val].to_numpy()

scaler = StandardScaler().fit(X_train)  # Normalizar los datos SOLO con train

# ---------- Entradas normalizadas ------------------------------- 
X_train = scaler.transform(X_train) 
X_test  = scaler.transform(X_test)
X_val   = scaler.transform(X_val)

# -------- Tensores ---------------------------------------------
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)

y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1) 
y_val_t   = torch.tensor(y_val,   dtype=torch.float32).unsqueeze(1)

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
    
# 5 ------------ Entrenamiento --------------------------------
N_ini = X_train_t.shape[1]
modelo = Semiconductor_Tc(N_ini)

loss_fun = nn.SmoothL1Loss()
opt = optim.AdamW(modelo.parameters(), lr=1e-4, weight_decay=1e-4)

# ---------------- Entrenamiento con early stopping ----------------
best_val = float('inf')
best_state = None
patience, wait = 30, 0  # si despues de 50 epocas no mejora -> break
max_epochs = 300        # Máximo número de épocas
batch_size = 50

for epoch in range(max_epochs):
    modelo.train()
    idx_perm = torch.randperm(len(X_train_t))
    for i in range(0, len(X_train_t), batch_size):
        b = idx_perm[i:i+batch_size]
        y_pred = modelo(X_train_t[b])
        loss = loss_fun(y_pred, y_train_t[b])
        opt.zero_grad(); loss.backward(); opt.step()
    
    modelo.eval()
    with torch.no_grad():
        y_val_pred = modelo(X_val_t)
        val_loss = loss_fun(y_val_pred, y_val_t).item()
    if (epoch+1) % 20 == 0 or epoch == 0:
        print(f"Época {epoch+1:3d} | Val loss : {val_loss:.4f}")

    if val_loss < best_val - 1e-4:
        best_val = val_loss
        best_state = {k: v.cpu().clone() for k, v in modelo.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping en época {epoch+1}")
            break

if best_state is not None:
    modelo.load_state_dict(best_state)
modelo.eval()

# 6 --------------- Métricas ----------------------------------
def Metricas(y_true_t, y_pred_t):
    y_true = y_true_t
    y_pred = y_pred_t
    y_np    = y_true.cpu().numpy().ravel()
    yhat_np = y_pred.cpu().numpy().ravel()

    mse  = float(np.mean((y_np - yhat_np)**2))
    mae  = float(np.mean(np.abs(y_np - yhat_np)))
    rmse = float(np.sqrt(mse))
    r2   = float(r2_score(y_np, yhat_np))
    return y_true, y_pred, mse, mae, rmse, r2

# --------- Evaluación del modelo ----------------------------
with torch.no_grad():
    z_val   = modelo(X_val_t)
    z_test  = modelo(X_test_t)
    z_train = modelo(X_train_t)
    
ytr_true,  z_train, mse_tr, mae_tr, rmse_tr, r2_tr = Metricas(y_train_t, z_train)
yval_true, z_val,   mse_vl, mae_vl, rmse_vl, r2_vl = Metricas(y_val_t,   z_val)
ytest_true, z_test, mse_tst, mae_tst, rmse_tst, r2_tst = Metricas(y_test_t,  z_test)

print("\n[Métricas en Kelvin]")
print("Train -> MSE:{:.3f}  MAE:{:.3f}  RMSE:{:.3f}  R^2:{:.3f}".format(mse_tr, mae_tr, rmse_tr, r2_tr))
print("Valid -> MSE:{:.3f}  MAE:{:.3f}  RMSE:{:.3f}  R^2:{:.3f}".format(mse_vl, mae_vl, rmse_vl, r2_vl))
print("Test  -> MSE:{:.3f}  MAE:{:.3f}  RMSE:{:.3f}  R^2:{:.3f}".format(mse_tst, mae_tst, rmse_tst, r2_tst))

#  7 ------- gráficos de resultados 
def mostrar_resultados(y_true_t, y_pred_t, titulo="Conjunto", idx_orig=None):
    y_true = y_true_t.cpu().numpy().ravel()
    y_pred = y_pred_t.cpu().numpy().ravel()
    resid = y_pred - y_true

    tabla = pd.DataFrame({
        "Tc_real":  y_true,
        "Tc_pred":  y_pred,
        "Error":    resid
    })
    if idx_orig is not None:
        tabla.insert(0, "idx_original", idx_orig[:len(tabla)])

    print(f"\n=== {titulo}: primeros 15 ===")
    print(tabla.head(15).round(3))
    
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.figure(figsize=(5.2,5.2))
    plt.scatter(y_true, y_pred, s=10, alpha=0.5)
    plt.plot(lims, lims, '--', lw=1)
    plt.xlim(lims); plt.ylim(lims)
    plt.xlabel("Tc real (K)"); plt.ylabel("Tc predicho (K)")
    plt.title(f"Paridad — {titulo}")
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(5.8,3.8))
    plt.hist(resid, bins=50, alpha=0.9)
    plt.xlabel("Residual (K)"); plt.ylabel("Frecuencia")
    plt.title(f"Residuales — {titulo}")
    plt.tight_layout(); plt.show()

mostrar_resultados(y_train_t, z_train, "Entrenamiento", idx_orig=Ind_train)
mostrar_resultados(y_val_t,   z_val,   "Validación",    idx_orig=Ind_val)
mostrar_resultados(y_test_t,  z_test,  "Test",          idx_orig=Ind_test)

print(f'Datos de train = {len(y_train)}')
print(f'Datos de test  = {len(y_test)}')
print(f'Datos de val   = {len(y_val)}')

test      = y_test_t.cpu().numpy().ravel()
test_pre  = z_test.cpu().numpy().ravel()

# ------------------------------------------------------------------------
# OPCIONAL: GUARDAR / CARGAR MODELO Y SCALER (DESCOMENTA PARA USAR)
# ------------------------------------------------------------------------
# # Guardar pesos del modelo (state_dict) y scaler con pickle:
# import pickle, os
# os.makedirs("modelos_tc", exist_ok=True)
# torch.save(modelo.state_dict(), "modelos_tc/modelo_tc_state.pt")
# with open("modelos_tc/scaler_tc.pkl", "wb") as f:
#     pickle.dump(scaler, f)
# print("Modelo y scaler guardados en carpeta 'modelos_tc'.")

# # Para cargar más tarde:
# # 1) reconstruir la misma arquitectura
# # modelo_cargado = Semiconductor_Tc(N_ini)
# # modelo_cargado.load_state_dict(torch.load("modelos_tc/modelo_tc_state.pt", map_location="cpu"))
# # modelo_cargado.eval()
# # 2) cargar scaler
# # with open("modelos_tc/scaler_tc.pkl", "rb") as f:
# #     scaler_cargado = pickle.load(f)
# # 3) usar: X_new_std = scaler_cargado.transform(X_new); y_hat = modelo_cargado(torch.tensor(X_new_std, dtype=torch.float32))





