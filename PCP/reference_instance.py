"""
Instancia 3SAT de referencia compartida por los cuatro experimentos de
3.6.1-3.6.4: K_PAIRS contradicciones independientes (x_i) AND (NOT x_i),
cada una codificada como dos clausulas 3SAT con el literal repetido.

Por que esta familia y no otra:
- Es insatisfactible por construccion para cualquier K_PAIRS, y el valor
  optimo es exactamente 1/2 (cada par contradictorio fuerza a perder
  exactamente una de sus dos clausulas), independientemente del tamano.
  Esto evita la dilucion del gap inicial que se observa con una unica
  cadena larga de implicaciones (donde unsat_0 ~ 1/n se diluye aun mas
  por las aristas nulas de relleno de degree_reduction).
- Escala n libremente sin afectar el alfabeto por nodo: degree_reduction
  con use_expander_cloud=True regulariza siempre al mismo grado d
  (aqui d=9), por lo que el alfabeto que introduce power_graph(t=1)
  (W_orig^(2d)) no depende de K_PAIRS. Solo crece linealmente el numero
  total de variables.
- Se usa una nube expansora real (use_expander_cloud=True) en vez del
  ciclo simple: un ciclo no es un buen expansor (autovalor cercano a 1),
  y el Lema de Amplificacion de Brecha exige expansion para amplificar
  en pocos pasos de potenciacion. Con el ciclo simple, unsat(phi^t)
  crece de forma casi lineal y muy lenta incluso para t grande.
"""
K_PAIRS = 100  # numero de variables / pares contradictorios
C_CLOUD = 3    # grado interno de la nube expansora de Sipser-Spielman


def reference_clauses(k=K_PAIRS):
    return [[i, i, i] for i in range(1, k + 1)] + [[-i, -i, -i] for i in range(1, k + 1)]


def reference_best_assignment(k=K_PAIRS):
    # Asignacion optima: todas las variables a 1. Satisface exactamente
    # la mitad de las 2k clausulas (las "(x_i)"), viola la otra mitad
    # (las "(NOT x_i)") => unsat(phi) = 1/2.
    return {i: 1 for i in range(1, k + 1)}
