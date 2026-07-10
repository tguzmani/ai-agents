"""Implementaciones de quicksort para probar el chunking de código.

Este archivo existe únicamente como material de prueba para los text splitters
orientados a código (por ejemplo RecursiveCharacterTextSplitter.from_language).
Contiene varias funciones y una clase para que haya distintos límites naturales
donde cortar.
"""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")


def quicksort(items: list[T]) -> list[T]:
    """Quicksort funcional y sencillo (no in-place).

    Elige el primer elemento como pivote y particiona el resto en menores y
    mayores mediante list comprehensions. Es la versión más legible, ideal
    para entender el algoritmo, aunque crea listas nuevas en cada llamada.
    """
    if len(items) <= 1:
        return items

    pivot = items[0]
    rest = items[1:]
    smaller = [x for x in rest if x < pivot]
    larger = [x for x in rest if x >= pivot]

    return quicksort(smaller) + [pivot] + quicksort(larger)


def quicksort_inplace(items: list[T], low: int = 0, high: int | None = None) -> list[T]:
    """Quicksort in-place clásico con el esquema de partición de Lomuto.

    Ordena la lista sobre sí misma sin crear listas auxiliares, por lo que es
    más eficiente en memoria que la versión funcional. Modifica y retorna la
    misma lista recibida.
    """
    if high is None:
        high = len(items) - 1

    if low < high:
        pivot_index = _partition(items, low, high)
        quicksort_inplace(items, low, pivot_index - 1)
        quicksort_inplace(items, pivot_index + 1, high)

    return items


def _partition(items: list[T], low: int, high: int) -> int:
    """Partición de Lomuto: coloca el pivote en su posición final ordenada.

    Toma el último elemento como pivote y reorganiza el subarreglo de forma
    que todos los menores queden a la izquierda y los mayores a la derecha,
    devolviendo el índice definitivo del pivote.
    """
    pivot = items[high]
    i = low - 1

    for j in range(low, high):
        if items[j] <= pivot:
            i += 1
            items[i], items[j] = items[j], items[i]

    items[i + 1], items[high] = items[high], items[i + 1]
    return i + 1


class QuickSorter:
    """Envoltorio orientado a objetos alrededor del algoritmo quicksort.

    Permite configurar si el orden es ascendente o descendente y lleva la
    cuenta de cuántas comparaciones se realizaron en la última ordenación,
    útil para experimentar y comparar rendimiento.
    """

    def __init__(self, descending: bool = False) -> None:
        self.descending = descending
        self.comparisons = 0

    def sort(self, items: list[T]) -> list[T]:
        """Ordena una copia de la lista y retorna el resultado."""
        self.comparisons = 0
        result = self._sort(list(items))
        return result[::-1] if self.descending else result

    def _sort(self, items: list[T]) -> list[T]:
        if len(items) <= 1:
            return items

        pivot = items[len(items) // 2]
        smaller, equal, larger = [], [], []

        for x in items:
            self.comparisons += 1
            if x < pivot:
                smaller.append(x)
            elif x > pivot:
                larger.append(x)
            else:
                equal.append(x)

        return self._sort(smaller) + equal + self._sort(larger)


if __name__ == "__main__":
    datos = [9, 3, 7, 1, 8, 2, 6, 5, 4, 0]
    print("original:  ", datos)
    print("funcional: ", quicksort(datos))
    print("in-place:  ", quicksort_inplace(list(datos)))
    print("clase desc:", QuickSorter(descending=True).sort(datos))
