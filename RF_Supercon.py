# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 13:47:35 2025

@author: giorn
"""
import numpy as np
import pandas as pd
import Funciones_Superconductores as FSC 

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
import inverse_desing as ID #ChemDomain, build_domain_from_dataset
from scipy.optimize import differential_evolution
import sys
import joblib
# Hola
# ===========================================================
CSV_PATH = "Data_base_supercond.csv"

X_desc = pd.read_csv("X_desc_magpie.csv")
y_desc = pd.read_csv("y_Tc.csv").iloc[:, 0]



# FUNCIÓN 1: ABS + NORMALIZACIÓN
# ===========================================================

Xn,yn,df = FSC.Normalizar(CSV_PATH)



# FUNCIÓN 2: ELIMINAR COMPOSICIONES REPETIDAS 
# ===========================================================
X,y = FSC.Eliminar_Comp_repetidas(Xn,yn)
print(list(X.columns))
print("Ag" in X.columns)
print

sys.exit()

#FSC.histograma(y,"T(K)")
print(X.shape)
print(max(y))


idx = np.arange(len(X))
#sys.exit()

idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42)

# SPLIT DE LOS DATOS  de composiciones
X_train = X.iloc[idx_train]
X_test  = X.iloc[idx_test]
y_train = y.iloc[idx_train]
y_test  = y.iloc[idx_test]

# Aplicas LOS MISMOS índices a descriptores
X_train_desc = X_desc.iloc[idx_train]
X_test_desc  = X_desc.iloc[idx_test]
y_train_desc = y_desc.iloc[idx_train]
y_test_desc  = y_desc.iloc[idx_test]


        
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
        

rf_desc = Pipeline(steps=[
                ("imp", SimpleImputer(strategy="median")),
                ("rf", RandomForestRegressor(
                    n_estimators=600,
                    max_depth=None,
                    min_samples_leaf=5,
                    max_features=0.2,
                    bootstrap=True,
                    oob_score=True,        
                    n_jobs=-1,
                    random_state=20,
                    criterion="squared_error",
                ))
            ])
        
# ============================================================

# ENTRENAMIENTO Y R² TEST Composicion 
# ============================================================
rf.fit(X_train, y_train)
r2_test = rf.score(X_test, y_test)
print(f"R² (test): {r2_test:.4f}")

# ENTRENAMIENTO Y R² TEST descriptores
# ============================================================
rf_desc.fit(X_train_desc, y_train_desc)
r2_test_desc = rf_desc.score(X_test_desc, y_test_desc)
print(f"R² (test descp): {r2_test_desc:.4f}")

# Predicciones 
y_pred_train = rf.predict(X_train)
y_pred_test  = rf.predict(X_test)

y_pred_train_desc = rf_desc.predict(X_train_desc)
y_pred_test_desc = rf_desc.predict(X_test_desc)





# === NUEVO: filtrar y graficar solo los "buenos" ===
# 6) Filtrar “mejores” por tolerancia de error absoluto
tol = 5  # ajusta a tu criterio
X_new_tr, y_new_tr, _, idx_good_tr = FSC.Filtro_mejores(X_train, y_train, y_pred_train, tol)
X_new_te, y_new_te, _ , idx_good_te= FSC.Filtro_mejores(X_test,  y_test,  y_pred_test,  tol)

# Filtras descriptores usando ESOS MISMOS índices
X_new_tr_desc = X_train_desc.iloc[idx_good_tr]
y_new_tr_desc = y_train_desc.iloc[idx_good_tr]

X_new_te_desc = X_test_desc.iloc[idx_good_te]
y_new_te_desc = y_test_desc.iloc[idx_good_te]

# 7) Unir y reentrenar desde cero
X_retrain = pd.concat([X_new_tr, X_new_te], axis=0).reset_index(drop=True)
y_retrain = pd.concat([y_new_tr, y_new_te], axis=0).reset_index(drop=True)

X_retrain_desc = pd.concat([X_new_tr_desc, X_new_te_desc], axis=0)
y_retrain_desc = pd.concat([y_new_tr_desc, y_new_te_desc], axis=0)

rf2 = clone(rf)               # nuevo modelo “en blanco”
rf2.fit(X_retrain, y_retrain) # reentrenado SOLO con “mejores”
score_r2=rf2.score(X_retrain,y_retrain)
print(f"R² (retrain): {score_r2:.4f}")

rf_desc2 = clone(rf_desc)
rf_desc2.fit(X_retrain_desc, y_retrain_desc)
print("R2 retrain desc:", rf_desc2.score(X_retrain_desc, y_retrain_desc))

y_new_pred = rf2.predict(X_retrain)
y_new_pred_des = rf_desc2.predict(X_retrain_desc)

# FSC.histograma(y_new_pred)
print(max(y_new_pred))



print(type(X_train))
print(X_train.shape)
print(len(X_train.columns))
print(X_train.columns[:10])
#joblib.dump(rf2,"models/rf_model.pkl")
#X_retrain.to_csv("data/X_data_clean.csv", index=False)
#y_retrain.to_csv("data/y_data_clean.csv", index=False)
joblib.dump(X_new_tr, "data/X_train.pkl")
joblib.dump(X_new_te, "data/X_test.pkl")
joblib.dump(y_new_tr, "data/y_train.pkl")
joblib.dump(y_new_te, "data/y_test.pkl")
sys.exit()

