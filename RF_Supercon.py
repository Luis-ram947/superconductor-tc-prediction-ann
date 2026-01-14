# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 13:47:35 2025

@author: giorn
"""
import numpy as np
import pandas as pd
import Funciones_Superconductores as FSC # Normalizar,Eliminar_Comp_repetidas,mostrar_resultados, Filtro_mejores

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone

# Hola
# ===========================================================
CSV_PATH = r"C:\Users\giorn\OneDrive\Escritorio\Todas las carpetas\Python\Super_cond\Data_base_supercond.csv"


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


        
# 4) Pipeline RF (imputación por seguridad + RF)
#=============================================================
rf = Pipeline(steps=[
                ("imp", SimpleImputer(strategy="median")),
                ("rf", RandomForestRegressor(
                    n_estimators=600,
                    max_depth=None,
                    min_samples_leaf=4,
                    max_features=0.5,
                    bootstrap=True,
                    oob_score=True,        
                    n_jobs=-1,
                    random_state=20,
                    criterion="squared_error",
                ))
            ])
        
# =============================================================

# ENTRENAMIENTO Y R² TEST 
# =============================================================
rf.fit(X_train, y_train)
r2_test = rf.score(X_test, y_test)
print(f"R² (test): {r2_test:.4f}")

# Predicciones 
y_pred_train = rf.predict(X_train)
y_pred_test  = rf.predict(X_test)

#GRAFICOS DE PARIDAD Y RESIDUALES 
#==============================================================

FSC.mostrar_resultados(y_train, y_pred_train, "train")
FSC.mostrar_resultados(y_test,  y_pred_test,  "test")



# === NUEVO: filtrar y graficar solo los "buenos" ===
# 6) Filtrar “mejores” por tolerancia de error absoluto
tol = 4  # ajusta a tu criterio
X_new_tr, y_new_tr, _ = FSC.Filtro_mejores(X_train, y_train, y_pred_train, tol)
X_new_te, y_new_te, _ = FSC.Filtro_mejores(X_test,  y_test,  y_pred_test,  tol)

# 7) Unir y reentrenar desde cero
X_retrain = pd.concat([X_new_tr, X_new_te], axis=0).reset_index(drop=True)
y_retrain = pd.concat([y_new_tr, y_new_te], axis=0).reset_index(drop=True)

rf2 = clone(rf)               # nuevo modelo “en blanco”
rf2.fit(X_retrain, y_retrain) # reentrenado SOLO con “mejores”
score_r2=rf2.score(X_retrain,y_retrain)
print(f"R² (retrain): {score_r2:.4f}")

y_new_pred = rf2.predict(X_retrain)

FSC.mostrar_resultados(y_retrain, y_new_pred,"retrain")





