"""
Serie de Taylor / Maclaurin
---------------------------
El usuario ingresa:
  f(x)  : función (sintaxis Python/SymPy, ej: exp(x), sin(x), log(1+x), 1/(1-x))
  a     : punto de expansión (a = 0 -> Maclaurin, a != 0 -> Taylor)
  x     : punto donde se evalúa la función
  n     : número de términos

Salida:
  - Tabla: iteración, término, sumatoria, error absoluto, error relativo
  - Gráficas: iteración vs término, iteración vs sumatoria,
              iteración vs error absoluto y relativo

Dependencias: python -m pip install sympy numpy matplotlib pandas
"""

import math
import re
import sympy as sp
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                        implicit_multiplication_application, convert_xor)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X = sp.Symbol("x")


def parsear_funcion(texto: str) -> sp.Expr:
    """Convierte el texto del usuario en una expresión simbólica (tolerante a errores de escritura).

    Acepta: signo menos Unicode (−), espacios dentro de números ("0. 333"),
    coma decimal (0,333), ^ como potencia y multiplicación implícita (100x, sin(x)cos(x)).
    """
    texto = texto.replace("−", "-").replace("–", "-").replace("×", "*").replace("÷", "/")
    texto = re.sub(r"\s+", "", texto)                 # quita espacios
    texto = re.sub(r"(?<=\d),(?=\d)", ".", texto)     # 0,333 -> 0.333
    trans = standard_transformations + (implicit_multiplication_application, convert_xor)
    return parse_expr(texto, local_dict={"x": X, "e": sp.E, "ln": sp.log, "pi": sp.pi},
                      transformations=trans)


def leer_numero(mensaje: str) -> float:
    """Pide un número (acepta pi, 1/2, 0,5, etc.) y repite si hay error."""
    while True:
        try:
            return float(sp.N(parsear_funcion(input(mensaje))))
        except Exception:
            print("  Valor no válido, intenta de nuevo (ej: 0, 0.5, 1/2, pi).")


def leer_funcion() -> sp.Expr:
    """Pide la función y repite si no se puede interpretar."""
    while True:
        try:
            f = parsear_funcion(input("f(x) = "))
            if f.free_symbols - {X}:
                raise ValueError("Solo se permite la variable x")
            return f
        except Exception as e:
            print(f"  No se pudo interpretar la función ({e}). Ejemplo: 0.333*exp(-100*x)*sin(300*x)")


PREC = 60  # dígitos de precisión interna (evita pérdida por cancelación numérica)


def serie_taylor(f: sp.Expr, a: float, x0: float, n: int) -> pd.DataFrame:
    """Calcula las n primeras iteraciones de la serie de Taylor de f en a, evaluada en x0.

    Los cálculos internos usan alta precisión; la tabla se entrega en float.
    """
    a_s = sp.nsimplify(a, rational=True)
    x_s = sp.nsimplify(x0, rational=True)
    valor_real = sp.N(f.subs(X, x_s), PREC)
    derivada = f
    suma = sp.Float(0, PREC)
    filas = []

    for k in range(n):
        coef = sp.N(derivada.subs(X, a_s), PREC)                 # f^(k)(a)
        termino = coef / sp.factorial(k) * sp.N((x_s - a_s) ** k, PREC)
        suma += termino

        err_abs = abs(valor_real - suma)
        err_rel = err_abs / abs(valor_real) if valor_real != 0 else None

        filas.append({
            "Iteración": k + 1,
            "Orden (k)": k,
            "f^(k)(a)": float(coef),
            "Término": float(termino),
            "Sumatoria": float(suma),
            "Error absoluto": float(err_abs),
            "Error relativo": float(err_rel) if err_rel is not None else np.nan,
            "Error relativo (%)": float(err_rel * 100) if err_rel is not None else np.nan,
        })
        derivada = sp.diff(derivada, X)                          # siguiente derivada

    df = pd.DataFrame(filas)
    df.attrs["valor_real"] = float(valor_real)
    return df


def graficar(df: pd.DataFrame, f: sp.Expr, a: float, x0: float, guardar: str | None = None):
    """Genera las tres gráficas pedidas."""
    tipo = "Maclaurin" if a == 0 else f"Taylor (a = {a})"
    valor_real = df.attrs["valor_real"]
    it = df["Iteración"]

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{tipo} de f(x) = {f}   |   x = {x0}   |   valor real = {valor_real:.10g}",
                 fontsize=12)

    # 1) iteración vs término
    ax[0].plot(it, df["Término"], "o-", color="tab:blue")
    ax[0].axhline(0, color="gray", lw=0.8)
    ax[0].set(title="Iteración vs Término", xlabel="Iteración", ylabel="Valor del término")

    # 2) iteración vs sumatoria
    ax[1].plot(it, df["Sumatoria"], "o-", color="tab:green", label="Sumatoria")
    ax[1].axhline(valor_real, color="red", ls="--", label="Valor real")
    ax[1].set(title="Iteración vs Sumatoria", xlabel="Iteración", ylabel="Suma parcial")
    ax[1].legend()

    # 3) iteración vs errores (absoluto y relativo)
    ea = df["Error absoluto"].replace(0, np.nan)
    er = df["Error relativo"].replace(0, np.nan)
    ax[2].semilogy(it, ea, "o-", color="tab:red", label="Error absoluto")
    ax[2].semilogy(it, er, "s--", color="tab:orange", label="Error relativo")
    ax[2].set(title="Iteración vs Errores (escala log)", xlabel="Iteración", ylabel="Error")
    ax[2].legend()
    ax[2].grid(True, which="both", alpha=0.3)

    for axis in ax[:2]:
        axis.grid(True, alpha=0.3)
        axis.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax[2].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.tight_layout()
    if guardar:
        fig.savefig(guardar, dpi=150)
    plt.show()


def main():
    print("=== Serie de Taylor / Maclaurin ===")
    print("Ejemplos: exp(x), sin(x), log(1+x), 1/(1-x), x^3+2x, 0.333*exp(-100*x)*sin(300*x)")
    f = leer_funcion()
    a = leer_numero("a (punto de expansión, 0 = Maclaurin) = ")
    x0 = leer_numero("x (punto a evaluar) = ")
    while True:
        try:
            n = int(input("Número de términos = "))
            if n >= 1:
                break
        except ValueError:
            pass
        print("  Ingresa un entero mayor o igual a 1.")

    df = serie_taylor(f, a, x0, n)

    tipo = "Maclaurin" if a == 0 else "Taylor"
    print(f"\nSerie de {tipo} de f(x) = {f} alrededor de a = {a}, evaluada en x = {x0}")
    print(f"Valor real f({x0}) = {df.attrs['valor_real']:.12g}\n")

    pd.set_option("display.float_format", lambda v: f"{v:.8g}")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", None)
    print(df.to_string(index=False))

    df.to_csv("tabla_taylor.csv", index=False)
    print("\nTabla guardada en tabla_taylor.csv")

    graficar(df, f, a, x0, guardar="graficas_taylor.png")


if __name__ == "__main__":
    main()