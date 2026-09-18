# CameraCaptor

Panel local para controlar una cámara IP mediante RTSP, HTTP y ONVIF.

Incluye:

- visualización del vídeo RTSP;
- detección de personas con YOLO11 nano;
- aviso de voz por el altavoz de la cámara;
- control independiente de iluminación y visión nocturna;
- movimiento PTZ mediante botones o teclado.

## Instalación

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Las credenciales se guardan localmente en `claves.txt`, archivo excluido del
repositorio. El modelo `yolo11n.pt` también se mantiene fuera del repositorio y
se descarga automáticamente si no está presente.

## Ejecución

```powershell
.\.venv\Scripts\python control_panel.py --credentials-file .\claves.txt --host 192.168.100.109
```

El umbral de ausencia previo a un nuevo aviso de persona puede ajustarse entre
10 y 15 segundos:

```powershell
.\.venv\Scripts\python control_panel.py --credentials-file .\claves.txt --host 192.168.100.109 --idle-seconds 12
```

## Funcionamiento

YOLO procesa solamente la clase `person`. El modelo se prepara al arrancar y
analiza hasta cinco fotogramas por segundo. Después del intervalo configurado
sin personas, dos detecciones positivas consecutivas disparan una vez la frase
«Se ha identificado un humano». No vuelve a hablar mientras la persona continúe
presente.

El TTS se sintetiza con Windows SAPI y se transmite como audio G.711 μ-law por
el backchannel RTSP. Después de un periodo sin utilizar el audio, el programa
reactiva primero la sesión de la cámara para evitar que el primer mensaje se
pierda.

Los focos y el modo de imagen tienen controles separados. El firmware de esta
cámara no permite garantizar focos blancos encendidos mientras permanece en
modo diurno.

Ultralytics se distribuye bajo AGPL-3.0 y ofrece una licencia Enterprise. Debe
revisarse su licencia antes de comercializar o desplegar este prototipo como
software cerrado.
