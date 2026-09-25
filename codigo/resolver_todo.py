"""
RESOLVER_TODO.PY
Corre, en un solo paso, las dos mitades del TP:
  1. replicar_tp.py  -> replicación de los 74 ítems originales (panel,
     Argentina, Wicksell, Hayek, moneda por país).
  2. estimar_paises.py -> extensión internacional (exceso de oferta de
     dinero y transmisión, los siete países).

Cada script corre en su propio espacio de nombres, así que si los dos
definen una función con el mismo nombre (por ejemplo 'mensual' o 'fred')
no se pisan entre sí. Necesita 'replicar_tp.py' y 'estimar_paises.py' en
la misma carpeta, y tp_datos/ ya con todo adentro (corré descargar_datos.py
primero si todavía no bajaste la parte internacional).
"""
import subprocess, sys, traceback


def instalar(*paquetes):
    faltan = []
    for p in paquetes:
        try:
            __import__(p)
        except ImportError:
            faltan.append(p)
    if faltan:
        print(f'  instalando: {", ".join(faltan)} ...')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + faltan, check=False)


instalar('linearmodels', 'statsmodels', 'openpyxl', 'requests')

print('\n' + '#' * 100)
print('# PARTE 1 — REPLICACIÓN DEL TP ORIGINAL (74 ítems)')
print('#' * 100)
ns1 = {}
try:
    exec(open('replicar_tp.py').read(), ns1)
    if 'rep' not in ns1 and 'replicar' in ns1:
        ns1['rep'] = ns1['replicar']()
except Exception:
    print('\n  [!] La parte 1 (replicación) terminó con un error:')
    traceback.print_exc()

print('\n\n' + '#' * 100)
print('# PARTE 2 — EXTENSIÓN INTERNACIONAL (siete países)')
print('#' * 100)
ns2 = {}
try:
    exec(open('estimar_paises.py').read(), ns2)
except Exception:
    print('\n  [!] La parte 2 (extensión internacional) terminó con un error:')
    traceback.print_exc()

print('\n\nListo. Resultados de la parte 1 en la variable de esa celda '
      '(replicacion_tp.csv en disco); de la parte 2, en '
      'cinco_paises_veredictos.csv.')
