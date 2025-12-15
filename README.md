# GAPlanner - engineering thesis

Aplikacja desktopowa do rozwiązywania problemu **VRPTW (Vehicle Routing Problem with Time Windows)** z użyciem **algorytmu genetycznego (GA)**.

## Funkcje
- Wczytywanie instancji VRPTW z pliku **CSV**
- Konfiguracja parametrów GA z poziomu GUI:
  - `pop` (rozmiar populacji)
  - `gens` (liczba generacji)
  - `pc` (prawdopodobieństwo krzyżowania)
  - `pm` (prawdopodobieństwo mutacji)
  - `alpha`, `beta` (wagi kar w funkcji celu)
  - `max_vehicles` (limit pojazdów)
- Wizualizacja:
  - przebieg najlepszego fitnessu w kolejnych generacjach
  - wykres tras na płaszczyźnie (X, Y)
- Zapis wyników do folderu wyjściowego

## Jak działa aplikacja ? (w skrócie)
- GA optymalizuje **permutację klientów**.
- Następnie permutacja jest zamieniana na zestaw mnijeszych tras.
- Wyliczane jest dopasownaie opierające się o metryki:
  - dystansu 
  - przeładowania 
  - spóźnienia względem okien czasowych 
  - ilości użytych pojazdów
- Aplikacja wykonuje działanie do momentu przekroczenia ustawionego dopuszczalnego czasu działaniu lub liczby iteracji. 

## Wymagania
- Python 3.10+ (zalecane)
- Biblioteki :
  - `PySide6`
  - `matplotlib`
  - `numpy`
  - `pandas`

