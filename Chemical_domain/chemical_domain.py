
from itertools import product #Generar todas las combinaciones posibles entre varias listas.
import random
# Lista de elementos 
ELEMENTS = [
"Al","Am","As","Au","B","Ba","Be","Bi","Br","C","Ca","Cd","Ce","Cl","Cm",
"Co","Cr","Cs","Cu","Dy","Er","Eu","F","Fe","Ga","Gd","Ge","H","Hf","Hg","Ho",
"I","In","Ir","K","La","Li","Lu","Mg","Mn","Mo","N","Na","Nb","Nd","Ni","No",
"Np","O","Os","P","Pa","Pb","Pd","Po","Pr","Pt","Pu","Rb","Re","Rh","Ru","S",
"Sb","Sc","Se","Si","Sm","Sn","Sr","Ta","Tb","Tc","Te","Th","Ti","Tl","Tm",
"U","V","W","Y","Yb","Zn","Zr"
]

# Estados de oxidacion 

OXIDATION_STATES = {
"Ag":[1,2],
"Al":[3],
"Am":[3,4],
"As":[-3,3,5],
"Au":[1,3],
"B":[3],
"Ba":[2],
"Be":[2],
"Bi":[3,5],
"Br":[-1,1,3,5],
"C":[-4,2,4],
"Ca":[2],
"Cd":[2],
"Ce":[3,4],
"Cl":[-1,1,3,5,7],
"Cm":[3],
"Co":[2,3],
"Cr":[2,3,6],
"Cs":[1],
"Cu":[2,3],
"Dy":[3],
"Er":[3],
"Eu":[2,3],
"F":[-1],
"Fe":[2,3],
"Ga":[3],
"Gd":[3],
"Ge":[2,4],
"H":[1,-1],
"Hf":[4],
"Hg":[1,2],
"Ho":[3],
"I":[-1,1,3,5,7],
"In":[1,3],
"Ir":[3,4],
"K":[1],
"La":[3],
"Li":[1],
"Lu":[3],
"Mg":[2],
"Mn":[2,3,4,7],
"Mo":[2,4,6],
"N":[-3,3,5],
"Na":[1],
"Nb":[3,5],
"Nd":[3],
"Ni":[2,3],
"No":[2,3],
"Np":[3,4,5,6],
"O":[-2],
"Os":[4,6],
"P":[-3,3,5],
"Pa":[4,5],
"Pb":[2,4],
"Pd":[2,4],
"Po":[2,4,6],
"Pr":[3,4],
"Pt":[2,4],
"Pu":[3,4,5,6],
"Rb":[1],
"Re":[4,6,7],
"Rh":[3],
"Ru":[3,4],
"S":[-2,4,6],
"Sb":[3,5],
"Sc":[3],
"Se":[-2,4,6],
"Si":[4],
"Sm":[2,3],
"Sn":[2,4],
"Sr":[2],
"Ta":[5],
"Tb":[3,4],
"Tc":[4,7],
"Te":[-2,4,6],
"Th":[4],
"Ti":[2,3,4],
"Tl":[1,3],
"Tm":[3],
"U":[3,4,5,6],
"V":[2,3,4,5],
"W":[4,6],
"Y":[3],
"Yb":[2,3],
"Zn":[2],
"Zr":[4]
}
POSITIVE_ELEMENTS = {
"Al":[3],
"Am":[3,4],
"Au":[1,3],
"B":[3],
"Ba":[2],
"Be":[2],
"Bi":[3,5],
"Ca":[2],
"Cd":[2],
"Ce":[3,4],
"Cm":[3],
"Co":[2,3],
"Cr":[2,3,6],
"Cs":[1],
"Cu":[2,3],
"Dy":[3],
"Er":[3],
"Eu":[2,3],
"Fe":[2,3],
"Ga":[3],
"Gd":[3],
"Ge":[2,4],
"Hf":[4],
"Hg":[1,2],
"Ho":[3],
"In":[1,3],
"Ir":[3,4],
"K":[1],
"La":[3],
"Li":[1],
"Lu":[3],
"Mg":[2],
"Mn":[2,3,4,7],
"Mo":[2,4,6],
"Na":[1],
"Nb":[3,5],
"Nd":[3],
"Ni":[2,3],
"No":[2,3],
"Np":[3,4,5,6],
"Os":[4,6],
"Pa":[4,5],
"Pb":[2,4],
"Pd":[2,4],
"Po":[2,4,6],
"Pr":[3,4],
"Pt":[2,4],
"Pu":[3,4,5,6],
"Rb":[1],
"Re":[4,6,7],
"Rh":[3],
"Ru":[3,4],
"Sb":[3,5],
"Sc":[3],
"Si":[4],
"Sm":[2,3],
"Sn":[2,4],
"Sr":[2],
"Ta":[5],
"Tb":[3,4],
"Tc":[4,7],
"Th":[4],
"Ti":[2,3,4],
"Tl":[1,3],
"Tm":[3],
"U":[3,4,5,6],
"V":[2,3,4,5],
"W":[4,6],
"Y":[3],
"Yb":[2,3],
"Zn":[2],
"Zr":[4]
}
NEGATIVE_ELEMENTS = {
"O":[-2],
"F":[-1],
"S":[-2],
"Se":[-2],
"Te":[-2],
"Cl":[-1],
"Br":[-1],
"I":[-1]
}

VARIABLE_VALENCE_ELEMENTS = {
"As":[-3,3,5],
"C":[-4,2,4],
"H":[-1,1],
"N":[-3,3,5],
"P":[-3,3,5]
}

# Elementos activos por composicion 
MAX_ACTIVE = 6
MIN_ACTIVE = 2
MIN_FRAC_ACTIVE = 0.01
CHARGE_TOL = 1e-4


Min_cationes = 2
Max_cationes = 4
Min_aniones = 1
Max_aniones = 2

#chequeo de cargas 

def charge_neutrality(x):
    #x es una lista de fracciones de tamaño 85
    activos = []
    fraccion = []
    for i in range (len(x)):
        if x[i]>MIN_FRAC_ACTIVE:
            activos.append(ELEMENTS[i])
            fraccion.append(x[i])
    cargas_posibles = [OXIDATION_STATES[k] for k in activos]

    # variable para saber si encontramos neutralidad
    es_neutra = False
    mejor_carga = None
    mejor_valencia = None

    # probar todas las combinaciones posibles
    for combinacion in product(*cargas_posibles):

        carga_total = 0.0

        # calcular carga total
        for j in range(len(combinacion)):
            carga_total = carga_total + combinacion[j] * fraccion[j]

        # verificar neutralidad
        if abs(carga_total) < CHARGE_TOL:
            es_neutra = True
            mejor_carga = carga_total
            mejor_valencia = combinacion
            break

    # si no se encuentra neutralidad, guardar la menor carga encontrada
    if es_neutra == False:

        menor_error = 1e9

        for combinacion in product(*cargas_posibles):

            carga_total = 0.0

            for j in range(len(combinacion)):
                carga_total = carga_total + combinacion[j] * fraccion[j]

            if abs(carga_total) < menor_error:
                menor_error = abs(carga_total)
                mejor_carga = carga_total
                mejor_valencia = combinacion

    return es_neutra, activos, fraccion, mejor_valencia, mejor_carga

def repair(x):

    # evitar negativos primero
    for i in range(len(x)):
        if x[i] < 0:
            x[i] = 0.0

    # quitar fracciones muy pequeñas
    for i in range(len(x)):
        if x[i] < MIN_FRAC_ACTIVE:
            x[i] = 0.0

    # limitar numero de activos
    activos_idx = []
    for i in range(len(x)):
        if x[i] > 0:
            activos_idx.append(i)

    if len(activos_idx) > MAX_ACTIVE:
        activos_ordenados = sorted(activos_idx, key=lambda i: x[i], reverse=True)

        conservar = activos_ordenados[:MAX_ACTIVE]
        eliminar = activos_ordenados[MAX_ACTIVE:]

        for i in eliminar:
            x[i] = 0.0

    # renormalizar antes de corregir carga
    suma_total = sum(x)
    if suma_total > 0:
        for i in range(len(x)):
            x[i] = x[i] / suma_total

    # ahora revisar neutralidad
    neutral, elementos, fracciones, valencias, mejor_carga = charge_neutrality(x)

    if neutral == True:
        return x

    contribuciones = []
    for i in range(len(elementos)):
        contribuciones.append(fracciones[i] * valencias[i])

    if mejor_carga < 0:
        idx_local = contribuciones.index(min(contribuciones))
        elemento_sel = elementos[idx_local]
        valencia_sel = valencias[idx_local]

        suma_otros = 0.0
        for i in range(len(contribuciones)):
            if i != idx_local:
                suma_otros = suma_otros + contribuciones[i]

        if valencia_sel != 0:
            nueva_fraccion = -suma_otros / valencia_sel
            idx_global = ELEMENTS.index(elemento_sel)
            x[idx_global] = max(0.0, nueva_fraccion)

    if mejor_carga > 0:
        idx_local = contribuciones.index(max(contribuciones))
        elemento_sel = elementos[idx_local]
        valencia_sel = valencias[idx_local]

        suma_otros = 0.0
        for i in range(len(contribuciones)):
            if i != idx_local:
                suma_otros = suma_otros + contribuciones[i]

        if valencia_sel != 0:
            nueva_fraccion = -suma_otros / valencia_sel
            idx_global = ELEMENTS.index(elemento_sel)
            x[idx_global] = max(0.0, nueva_fraccion)

    # limpieza final
    for i in range(len(x)):
        if x[i] < MIN_FRAC_ACTIVE:
            x[i] = 0.0

    suma_total = sum(x)
    if suma_total > 0:
        for i in range(len(x)):
            x[i] = x[i] / suma_total

    return x

def is_feasible_composition(x):
    neutral, elementos, fracciones, valencias, mejor_carga = charge_neutrality(x)
    if not neutral:
        return False

    if len(elementos) > MAX_ACTIVE:
        return False

    if any(xi < 0 for xi in x):
        return False

    if abs(sum(x) - 1) > 1e-6:
        return False
    
    return True

def generate_valid_composition():
    while True:
        x = [0.0]*len(ELEMENTS)

        N_aniones = random.randint(Min_aniones,Max_aniones)
        N_cationes = random.randint(Min_cationes,Max_cationes)

        lista_aniones = list(NEGATIVE_ELEMENTS.keys())
        lista_cationes = list(POSITIVE_ELEMENTS.keys())
        
        #Seleccion de elementos 
        aniones = random.sample(lista_aniones,N_aniones)
        cationes = random.sample(lista_cationes,N_cationes)

        elementos_selec = aniones + cationes

        if len(elementos_selec) > MAX_ACTIVE:
            continue

        # generar fracciones aleatorias positivas
        fracciones_random = []
        for i in range(len(elementos_selec)):
            fracciones_random.append(random.random())
        # normalizar
        suma = sum(fracciones_random)
        fracciones = []
        for f in fracciones_random:
            fracciones.append(f/suma)

        # meter al vector x
        for i in range(len(elementos_selec)):
            elemento = elementos_selec[i]
            idx = ELEMENTS.index(elemento)
            x[idx] = fracciones[i]

        # reparar
        x = repair(x)

        # validar
        if is_feasible_composition(x) == True:
            return x


    

    

