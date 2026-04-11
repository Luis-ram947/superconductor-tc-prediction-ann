import chemical_domain as CD
import chemical_domain as CD

import chemical_domain as CD


x_neutra = [0.0]*len(CD.ELEMENTS)
x_no_neutra = [0.0]*len(CD.ELEMENTS)

# Caso neutro: Ba2CuO4
x_neutra[CD.ELEMENTS.index("Ba")] = 2/6
x_neutra[CD.ELEMENTS.index("Cu")] = 1/6
x_neutra[CD.ELEMENTS.index("O")]  = 3/6

# Caso no neutro: Ba2CuO2
x_no_neutra[CD.ELEMENTS.index("Ba")] = 2/5
x_no_neutra[CD.ELEMENTS.index("Cu")] = 1/5
x_no_neutra[CD.ELEMENTS.index("O")]  = 2/5

print("CASO NEUTRO")
print(CD.charge_neutrality(x_neutra))

print("\nCASO NO NEUTRO")
print(CD.charge_neutrality(x_no_neutra))

x_nueva =CD.repair(x_no_neutra)
print("CASO NO NEUTRO Reparado")
print(CD.charge_neutrality(x_nueva))

for i in range(5):
    x = CD.generate_valid_composition()
    print(CD.is_feasible_composition(x))
    print(sum(x))
    print(CD.charge_neutrality(x))
    print("-"*40)

print(len(x))
print(len(CD.ELEMENTS))