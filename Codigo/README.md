# Código

Esta carpeta contiene tres variantes del mismo algoritmo NEAT. Las tres comparten exactamente la misma estructura de carpetas, el mismo `Makefile` y la misma interfaz de ejecución; lo que cambia es el modelo de plasticidad.

| Carpeta   | Descripción |
|-----------|-------------|
| `NEAT/`   | NEAT estándar, sin plasticidad R-STDP (conexiones fijas tras la evolución). |
| `Fijo/`   | NEAT + R-STDP con los parámetros de plasticidad (`tau_c`, `a_plus`, `a_minus`, `tau_plus`, `tau_minus`) fijos durante toda la evolución. |
| `Especie/`| NEAT + R-STDP donde, además, los parámetros de plasticidad se mutan periódicamente (cada 5 generaciones) durante la evolución. |

Cada carpeta es autocontenida y se compila/ejecuta de forma independiente siguiendo los mismos pasos.

## Estructura de cada carpeta

```
<Especie|Fijo|NEAT>/
├── Makefile
├── NEAT              # ejecutable generado por `make`
├── annarchy.py        # entorno simulado (cartpole, acrobot, etc.) en ANNarchy
├── bin/                # objetos .o generados por la compilación
├── config/
│   └── config.cfg      # archivo de configuración por defecto
├── headers/
└── src/
```

## Requisitos

- `g++` con soporte C++11
- Python 3.10 con headers de desarrollo (`libpython3.10-dev` en Debian/Ubuntu)
- El paquete Python [ANNarchy](https://annarchy.readthedocs.io/) (usado desde `annarchy.py`)

## 1. Configuración (`config/config.cfg`)

Antes de compilar/ejecutar, revisa y ajusta `config/config.cfg` dentro de la carpeta que quieras usar (`Especie`, `Fijo` o `NEAT`). Es un archivo `clave=valor` con los hiperparámetros de la evolución:

```
keep=0.577
threshold=3.241
interSpeciesRate=0.000549
noCrossoverOff=0.329
probabilityWeightMutated=0.851
probabilityAddNodeSmall=0.0305
probabilityAddLink_small=0.0436
probabilityAddNodeLarge=0.0417
probabilityAddLink_Large=0.0789
largeSize=20
c1=0.53
c2=0.959
c3=0.306
numberGenomes=2
numberInputs=8
numberOutputs=2
evolutions=25
n_max=200
learningRate=5
inputWeights=110,150
weightsRange=-20,80
process_max=2
function=cartpole_ns
tau_c=0.0
a_minus=0.0
a_plus=0.0
tau_plus=0.0
tau_minus=0.0
tunable_params=no
```

- `function`: entorno a evaluar (`cartpole_ns`, `acrobot_ns`, entre otros definidos en `annarchy.py`).
- `numberInputs` / `numberOutputs`: deben coincidir con el entorno elegido (p. ej. `cartpole_ns` usa 8/2, `acrobot_ns` usa 12/3).
- `tau_c`, `a_plus`, `a_minus`, `tau_plus`, `tau_minus`: solo aplican a `Fijo` y `Especie` (parámetros de R-STDP). En `NEAT` se ignoran.
- Puedes fijar además `folder=results/trial-N` si quieres apuntar a una dirección específica.

Este archivo es el que se usa siempre que **no** se entreguen 18 parámetros por consola, durante la ejecución.

## 2. Compilar

Dentro de la carpeta elegida (`Especie`, `Fijo` o `NEAT`):

```bash
cd Codigo/NEAT      # o Codigo/Fijo, o Codigo/Especie
make clean          # limpia bin/, el ejecutable, __pycache__, annarchy/ y results/
make                # compila y genera el ejecutable ./NEAT
```


## 3. Ejecutar

El binario `./NEAT` acepta tres formas de utilizacion según la cantidad de argumentos, no usa flags con nombre, son por posicion:

### a) Sin argumentos — `./NEAT`

Usa `config/config.cfg` tal cual y corre el trial `0`. Los resultados y una copia del config usado quedan en `results/trial-0/`.

```bash
./NEAT
```

### b) Un argumento — `./NEAT N`

Igual que el caso anterior (lee `config/config.cfg`), pero corre/guarda bajo `results/trial-N/`. Útil para lanzar varios trials en paralelo con la misma configuración.

```bash
./NEAT 3
```

### c) 18 argumentos — `./NEAT p1 p2 ... p18`

Permite sobrescribir por consola los hiperparámetros evolutivos y de plasticidad sin tocar `config/config.cfg` (el resto de valores — `numberGenomes`, `numberInputs`, `numberOutputs`, `evolutions`, `n_max`, `learningRate`, `inputWeights`, `weightsRange`, `process_max`, `function`, `tunable_params`, se siguen leyendo desde `config/config.cfg`). Se genera `results/trial-<trialNumber>/config.cfg` con la unión de los parametros, esto fue utilizado durante el procedimiento de optimización con optuna

Deben entregarse **exactamente 18 valores**, en este orden:

| # | Parámetro | Config equivalente |
|---|-----------|---------------------|
| 1 | `keep` | `keep` |
| 2 | `threshold` | `threshold` |
| 3 | `probabilityInterSpecies` | `interSpeciesRate` |
| 4 | `probabilityNoCrossoverOff` | `noCrossoverOff` |
| 5 | `probabilityWeightMutated` | `probabilityWeightMutated` |
| 6 | `probabilityAddNodeSmall` | `probabilityAddNodeSmall` |
| 7 | `probabilityAddLink_small` | `probabilityAddLink_small` |
| 8 | `probabilityAddNodeLarge` | `probabilityAddNodeLarge` |
| 9 | `probabilityAddLink_Large` | `probabilityAddLink_Large` |
| 10 | `c1` | `c1` |
| 11 | `c2` | `c2` |
| 12 | `c3` | `c3` |
| 13 | `trialNumber` | define `results/trial-<N>` |
| 14 | `tau_c` | `tau_c` (R-STDP, `Fijo`/`Especie`) |
| 15 | `a_plus` | `a_plus` |
| 16 | `a_minus` | `a_minus` |
| 17 | `tau_plus` | `tau_plus` |
| 18 | `tau_minus` | `tau_minus` |

```bash
./NEAT 0.577 3.241 0.000549 0.329 0.851 0.0305 0.0436 0.0417 0.0789 0.53 0.959 0.306 0 5.69402 0.0417365 0.0165116 16.5692 33.9051
```



Cualquier otra cantidad de argumentos que no sea 0, 1 o 18, se trata como el caso a por defecto (config del archivo, trial 0)

## Resultados

Cada ejecución crea/actualiza `results/trial-<N>/` con:
- `config.cfg`: copia de la configuración efectivamente usada en esa corrida.
- `results.txt`: log de resultados de la evolución.

Al finalizar, el programa imprime en consola el mejor fitness obtenido (`Final fitness:`).
