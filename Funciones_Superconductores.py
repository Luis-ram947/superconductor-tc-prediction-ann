# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 14:06:23 2025

@author: giorn
"""

"""
Utility functions for preprocessing, feature generation, visualization,
and post-analysis of superconductor composition datasets.

This module includes:
- composition normalization
- duplicate removal
- filtering of best predictions
- descriptor generation with matminer
- visualization of Tc distributions and model performance
- chemical-space analysis with PCA and UMAP
"""

# =========================
# Standard scientific stack
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Materials informatics
# =========================
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty

# =========================
# Machine learning utilities
# =========================
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

# =========================
# Miscellaneous
# =========================
import ast
import umap


# ==========================================
# DATA PREPROCESSING AND COMPOSITION HANDLING
# ==========================================


def Normalizar (csv_name,skip_zero=True):
    """
    Load a dataset from CSV and normalize the compositional inputs row-wise.

    Parameters
    ----------
    csv_name : str
        Path to the CSV file. All columns except the last one are assumed
        to be compositional inputs, and the last column is the target.
    skip_zero : bool, optional
        If True, rows with zero total composition are removed before
        normalization.

    Returns
    -------
    Xn : pandas.DataFrame
        Row-wise normalized compositional input matrix.
    y : pandas.Series
        Target variable.
    df : pandas.DataFrame
        Original loaded dataframe.
    """
    # Load dataset
    df = pd.read_csv(csv_name, index_col=0)
    R,C=df.shape #Dimensiones de la matriz
    
    # Separate compositional inputs and target
    X = df.iloc[:,0:-1].abs().fillna(0.0)    
    y = df.iloc[:,-1]     
    
    # Normalize each row so that compositions sum to 1
    sums = X.sum(axis=1) #axis = 1 suma las columnas 
    if skip_zero :
        mask = sums >0 # True si sums es mayor a cero y false si sums es cero o menor
        X = X[mask]
        y = y[mask]
        sums = sums[mask]
    Xn = X.div(sums,axis=0) # axis = 0 
    return Xn,y,df

    


    
def Eliminar_Comp_repetidas (X,y,keep="first"):
    """
    Remove duplicated compositions after rounding.

    Parameters
    ----------
    X : pandas.DataFrame
        Compositional input matrix.
    y : pandas.Series or array-like
        Target values associated with X.
    keep : {"first", "last"}, optional
        Which duplicate entry to keep.

    Returns
    -------
    X_c : pandas.DataFrame
        Deduplicated compositions.
    y_c : pandas.Series
        Target values after duplicate removal.
    """
    df = X.copy()

    df["__y__"] = y.values if isinstance(y, pd.Series) else y

    # Round compositions to reduce floating-point duplication issues
    df = df.round(8).drop_duplicates(keep=keep)  
    X_c = df.drop(columns="__y__") #drop y columns
    y_c = df["__y__"].reset_index(drop=True) 
    return X_c,y_c
    
    
def Filtro_mejores (X,y_real,y_pred,tol,return_pandas=True,return_index=True):
    """
    Filter samples whose prediction error is within a given tolerance.

    Parameters
    ----------
    X : pandas.DataFrame or ndarray
        Input samples.
    y_real : pandas.Series or ndarray
        True target values.
    y_pred : ndarray
        Predicted target values.
    tol : float
        Absolute error tolerance.
    return_pandas : bool, optional
        If True and X is a DataFrame, return pandas objects.
    return_index : bool, optional
        If True, also return the selected positional indices.

    Returns
    -------
    X_best, y_best, y_best_p [, pos_best]
        Filtered samples, true values, predicted values, and optionally indices.
    """
    # Convert to arrays to calculate the mask by position
    X_is_df = isinstance(X, pd.DataFrame)
    y_is_series = isinstance(y_real, pd.Series)

    X_arr = X.values if X_is_df else np.asarray(X)
    y_arr = y_real.values if y_is_series else np.asarray(y_real)
    ypred_arr = np.asarray(y_pred)

    # Quick checks
    if ypred_arr.shape[0] != X_arr.shape[0] or y_arr.shape[0] != X_arr.shape[0]:
        raise ValueError("Dimensiones no coinciden entre X, y_real y y_pred.")

    # mask by position
    mask = np.abs(y_arr - ypred_arr) <= tol

    
    pos_best = np.where(mask)[0]

    # Select samples whose absolute prediction error is below the tolerance

    if return_pandas and X_is_df:
        X_best = X.iloc[pos_best]
        y_best = y_real.iloc[pos_best] if y_is_series else pd.Series(y_arr[pos_best], index=X_best.index, name="y")
        y_best_p = pd.Series(ypred_arr[pos_best], index=X_best.index, name="y_pred")
    else:
        X_best = X_arr[pos_best]
        y_best = y_arr[pos_best]
        y_best_p = ypred_arr[pos_best]

    if return_index:
        return X_best, y_best, y_best_p, pos_best
    return X_best, y_best, y_best_p
    


    
def composition_to_for(row,tol = 1e-6):
    """
    Convert a compositional row into a pymatgen Composition object.

    Small fractions below `tol` are ignored.
    """
    For_composition = { el:frac for el, frac in row.items() if frac > tol }
    return Composition(For_composition)

def X_descriptor (df):
    """
    Generate Magpie composition-based descriptors using matminer.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a 'composition' column with pymatgen
        Composition objects.

    Returns
    -------
    X_desc : pandas.DataFrame
        Descriptor matrix excluding the original composition column.
    """
    ep = ElementProperty.from_preset("magpie")
    df_feat = ep.featurize_dataframe(
        df,
        col_id = "composition",
        ignore_errors = True
    )   
    X_desc = df_feat.drop(columns = ["composition"])
    return X_desc



# =========================
# VISUALIZATION AND METRICS
# =========================



def histograma(y,a,b,ylabel = ""):
    """
    Plot a histogram of the target variable.
    """
    plt.figure()
    plt.hist(y, bins=a, alpha=b)
    plt.xlabel(ylabel); plt.ylabel("Frecuencia")
    plt.title(f"Histograma de temperatura")
    plt.tight_layout(); 

def residuos(y_true, y_pred, titulo="Conjunto"):
    """
    Plot the residual distribution for model predictions.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    titulo : str, optional
        Plot title suffix.
    """

    resid = y_pred - y_true

    posicion = np.arange(len(y_true))

    tabla = pd.DataFrame({
        "Tc_real":  y_true,
        "Tc_pred":  y_pred,
        "Error":    resid
    })
    plt.figure(figsize=(5.8,3.8))
    plt.hist(resid, bins=50, alpha=0.9)
    plt.xlabel("Residual (K)"); plt.ylabel("Frecuencia")
    plt.title(f"Residuales — {titulo}")
    plt.tight_layout(); 
    


def Paridad(y_true, y_pred, titulo="Conjunto"):
    """
    Plot a parity plot comparing true and predicted Tc values.

    Also prints the first 15 samples for quick inspection.
    """
    resid = y_pred - y_true

    posicion = np.arange(len(y_true))

    tabla = pd.DataFrame({
        "Tc_real":  y_true,
        "Tc_pred":  y_pred,
        "Error":    resid
    })


    print(f"\n=== {titulo}: primeros 15 ===")
    print(tabla.head(15).round(3))
    
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.figure(figsize=(5.2,5.2))
    plt.scatter(y_true, y_pred, s=10, alpha=0.5)
    plt.plot(lims, lims, '--', lw=1)
    plt.xlim(lims); plt.ylim(lims)
    plt.xlabel("Tc real(K)"); plt.ylabel("Tc prediccion (K)")
    plt.title(f"Paridad — {titulo}")
    plt.tight_layout(); 


def grafica_y_ordenada(y, titulo="Valores de y ordenados"):
    """
    Plot the sorted target values to inspect their distribution.
    """
    
    y = np.asarray(y).ravel()
    y_sorted = np.sort(y)
    
    plt.figure(figsize=(7,5))
    
    plt.plot(y_sorted)
    
    plt.xlabel("Índice ordenado")
    plt.ylabel("Tc (K)")
    plt.title(titulo)
    
    plt.grid(alpha=0.3)
    
   

def comportamiento_con_error(y_true, y_pred):
    """
    Plot sorted true and predicted values together with an absolute-error band.
    """
    
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    idx = np.argsort(y_true)
    
    y_true = y_true[idx]
    y_pred = y_pred[idx]
    
    error = np.abs(y_true - y_pred)
    
    plt.figure(figsize=(7,5))
    
    
    plt.plot(y_pred, label="Predicción")
    
    plt.fill_between(
        range(len(y_true)),
        y_pred-error,
        y_pred+error,
        alpha=0.2
    )
    plt.plot(y_true, label="Real")
    plt.legend()
    plt.xlabel("Índice ordenado")
    plt.ylabel("Tc (K)")
    
    plt.title("Comportamiento del modelo")


# ==================================
# CHEMICAL SPACE AND ELEMENT ANALYSIS
# ==================================
    


def superficie_3D(matriz):


    Z = matriz.values

    x = np.arange(Z.shape[1])
    y = np.arange(Z.shape[0])

    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(X, Y, Z, cmap="viridis")

    ax.set_xticks(x)
    ax.set_xticklabels(matriz.columns, rotation=45)

    ax.set_yticks(y)
    ax.set_yticklabels(matriz.index)

    ax.set_xlabel("Rango de Tc")
    ax.set_ylabel("Elemento")
    ax.set_zlabel("Frecuencia")

def heatmap_elementos_tc(
        X,
        y,
        bins_tc=None,
        umbral=1e-6,
        top_n=None,
        titulo="Frecuencia de elementos por rango de Tc"):
    """
    Plot a heatmap showing the relative frequency of element occurrence
    across Tc intervals.

    Parameters
    ----------
    X : pandas.DataFrame
        Composition matrix with elements as columns.
    y : array-like
        Critical temperature values.
    bins_tc : list, optional
        Temperature bin edges.
    umbral : float, optional
        Minimum fraction required to consider an element present.
    top_n : int, optional
        If given, only the most frequent elements are displayed.
    titulo : str, optional
        Plot title.
    """

    X = X.copy()
    y = pd.Series(np.asarray(y).ravel(), index=X.index)

    if bins_tc is None:
        bins_tc = [0, 20, 40, 60, 80, 100, 140]

    # Select most frequent elements if requested
    if top_n is not None:

        frecuencia_global = (X > umbral).sum(axis=0)

        top_elementos = (
            frecuencia_global
            .sort_values(ascending=False)
            .head(top_n)
            .index
        )

        X = X[top_elementos]

    # Split samples by Tc ranges

    grupos_tc = pd.cut(y, bins=bins_tc, include_lowest=True)

    elementos = X.columns.tolist()
    etiquetas_bins = [str(cat) for cat in grupos_tc.cat.categories]

    matriz = pd.DataFrame(0.0, index=elementos, columns=etiquetas_bins)
    
    # Compute relative element frequency within each Tc bin

    for cat in grupos_tc.cat.categories:

        idx = grupos_tc == cat
        X_bin = X.loc[idx]

        if len(X_bin) == 0:
            continue

        freq = (X_bin > umbral).sum(axis=0) / len(X_bin)

        matriz.loc[:, str(cat)] = freq.values


    plt.figure(figsize=(10, 8))

    im = plt.imshow(
        matriz.values,
        aspect='auto',
        origin='lower',
        cmap="viridis"
    )

    plt.xticks(range(len(matriz.columns)), matriz.columns, rotation=45)
    plt.yticks(range(len(matriz.index)), matriz.index)

    plt.xlabel("Rango de Tc (K)")
    plt.ylabel("Elemento")

    if top_n is not None:
        titulo = f"{titulo} (Top {top_n})"

    plt.title(titulo)

    plt.colorbar(im, label="Frecuencia de aparición")

    plt.tight_layout()

    return matriz

def frecuencia_elementos(X, umbral=1e-6, top_n=None, titulo="Frecuencia de aparición de elementos"):
    
    """
    Compute and plot the occurrence frequency of elements in the dataset.
    """
    
    # contar apariciones
    freq = (X > umbral).sum(axis=0)
    
    # ordenar
    freq = freq.sort_values(ascending=False)
    
    if top_n is not None:
        freq = freq.head(top_n)
    
    # graficar
    plt.figure(figsize=(10,6))
    
    plt.bar(freq.index, freq.values)
    
    plt.xticks(rotation=90)
    
    plt.xlabel("Elemento")
    plt.ylabel("Número de apariciones")
    
    if top_n is not None:
        plt.title(f"{titulo} (Top {top_n})")
    else:
        plt.title(titulo)
    
    plt.tight_layout()
    
    return freq

def calcular_knn(X_dataset, X_candidates, k=5):
    """
    Compute k-nearest-neighbor distances from candidate compositions
    to the reference dataset in composition space.
    """

    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X_dataset)

    distancias, indices = knn.kneighbors(X_candidates)

    return distancias

def composiciones_a_matriz_y_formula(df_candidatos, columnas_elementos, col_comp="composition", decimales_formula=3):
    """

    Convert a composition column stored as dictionaries into:
    1) a numerical matrix aligned with the original dataset columns
    2) a human-readable chemical formula string

    Parameters
    ----------
    df_candidatos : pd.DataFrame
        DataFrame that must contain a column with composition dictionaries
        stored as strings.
    columnas_elementos : list or pd.Index
        Columns of the original dataset in the correct order.
    col_comp : str
        Name of the column containing the composition.
    decimales_formula : int
        Number of decimal places used when constructing the chemical formula.

    Returns
    -------
    X_candidates : np.ndarray
        Numerical matrix of compositions.
    df_out : pd.DataFrame
        Copy of the original DataFrame with additional columns:
        - formula
    """

    columnas_elementos = list(columnas_elementos)
    X_candidates = []
    formulas = []

    for comp in df_candidatos[col_comp]:
        # convertir string -> dict
        comp_dict = ast.literal_eval(comp)

        # vector de composición
        vector = np.zeros(len(columnas_elementos), dtype=float)

        for i, elem in enumerate(columnas_elementos):
            vector[i] = comp_dict.get(elem, 0.0)

        X_candidates.append(vector)

        # fórmula química ordenada por fracción descendente
        comp_filtrada = {k: v for k, v in comp_dict.items() if v > 0}

        # ordenar por fracción de mayor a menor
        comp_ordenada = sorted(comp_filtrada.items(), key=lambda x: x[1], reverse=True)

        formula = ""
        for elem, frac in comp_ordenada:
            formula += f"{elem}{round(frac, decimales_formula)}"

        formulas.append(formula)

    X_candidates = np.array(X_candidates)

    df_out = df_candidatos.copy()
    df_out["formula"] = formulas

    return X_candidates, df_out

def PCA_graf (x_p, X_candidates):
    """
    Project dataset and candidate compositions into a 2D PCA space.
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(x_p.to_numpy())
    X_candidates_pca = pca.transform(X_candidates)
    plt.figure(figsize=(8, 6))

    plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        alpha=0.2,
        s=20,
        label="Dataset"
    )

    plt.scatter(
        X_candidates_pca[:, 0],
        X_candidates_pca[:, 1],
        s=80,
        label="GA candidates"
    )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA del espacio químico")
    plt.legend()
    plt.tight_layout()

def umap_dataset_y_candidatos(
    X_dataset,
    y_dataset,
    X_candidates,
    df_candidatos=None,
    n_neighbors=30,
    min_dist=0.05,
    titulo="UMAP del espacio químico"
):
    """
    Fit a UMAP representation on the reference dataset and project
    candidate compositions into the same embedding.

    Parameters
    ----------
    X_dataset : array-like
        Reference composition matrix.
    y_dataset : array-like
        Tc values used only for coloring the dataset points.
    X_candidates : array-like
        Candidate composition matrix.
    df_candidatos : pandas.DataFrame, optional
        Candidate dataframe to which UMAP coordinates will be appended.
    n_neighbors : int, optional
        UMAP local neighborhood size.
    min_dist : float, optional
        UMAP minimum distance parameter.
    titulo : str, optional
        Plot title.
    """
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        init="pca",
        random_state=42
    )

    X_umap = reducer.fit_transform(np.asarray(X_dataset))
    X_candidates_umap = reducer.transform(np.asarray(X_candidates))

    plt.figure(figsize=(8, 6))

    sc = plt.scatter(
        X_umap[:, 0],
        X_umap[:, 1],
        c=np.asarray(y_dataset).ravel(),
        alpha=0.35,
        s=20
    )

    plt.scatter(
        X_candidates_umap[:, 0],
        X_candidates_umap[:, 1],
        s=30,
        c="red",
        marker="o",
        label="GA candidates"
    )

    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.title(titulo)
    plt.colorbar(sc, label="Tc (K)")
    plt.legend()
    plt.tight_layout()
    

    if df_candidatos is not None:
        df_out = df_candidatos.copy()
        df_out["UMAP1"] = X_candidates_umap[:, 0]
        df_out["UMAP2"] = X_candidates_umap[:, 1]
        return reducer, df_out

    return reducer