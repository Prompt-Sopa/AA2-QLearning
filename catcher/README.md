# Lab 12 - Q-Learning
El objetivo de este laboratorio es entrenar agentes para resolver videojuegos sencillos.
## Preparacion del entorno
Creamos el entorno virual:
```
python3 -m venv env
```
Activamos el entorno:
```
source env/bin/activate
```
Instalamos dependencias:
```
pip3 install -r requirements.txt
```

## Videojuego Pong
Agente con acciones aleatorias:
```
python3 a_random-pong.py
```
Entrenamiento de agente con Q-Learning:
```
python3 b_train-pong.py
```
Test de agente con Q-Learning:
```
python3 c_test-pong.py
```