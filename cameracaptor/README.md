# CameraCaptor

Panel local para controlar una cámara IP mediante RTSP, HTTP y ONVIF.

Incluye:

- visualización del vídeo RTSP;
- detección de personas con YOLO11 nano;
- aviso de voz por el altavoz de la cámara;
- control independiente de iluminación y visión nocturna;
- movimiento PTZ mediante botones o teclado.

## Instalación

En Windows, instala Python 3 (probado con Python 3.13) y ejecuta una sola vez:

```powershell
.\instalar.bat
```

El branch incluye el modelo YOLO, los modelos faciales YuNet/SFace, la galería
de Brayan y `claves.txt`, tal como se requiere para esta cámara en la red local.
No hace falta instalar FFmpeg por separado: `requirements.txt` instala una copia
compatible mediante `imageio-ffmpeg`.

## Ejecución

```powershell
.\ejecutar_panel.bat
```

El script usa `192.168.10.143` por defecto. Para indicar otra IP:

```powershell
.\ejecutar_panel.bat 192.168.10.143
```

El umbral de ausencia previo a un nuevo aviso de persona puede ajustarse entre
10 y 15 segundos:

```powershell
.\.venv\Scripts\python control_panel.py --credentials-file .\claves.txt `
  --host 192.168.10.143 --faces-dir .\face_gallery --idle-seconds 12
```

## Funcionamiento

YOLO procesa solamente la clase `person`. El modelo se prepara al arrancar y
analiza hasta cinco fotogramas por segundo. Después del intervalo configurado
sin personas, dos detecciones positivas consecutivas habilitan un solo aviso.
Primero dice «Se ha identificado a alguien en la entrada». La llegada permanece
activa y, si después reconoce un rostro conocido, emite una segunda frase como
«Bienvenido Brayan». Ninguna de las dos frases se repite continuamente.

El TTS se sintetiza con Windows SAPI y se transmite como audio G.711 μ-law por
el backchannel RTSP. Después de un periodo sin utilizar el audio, el programa
reactiva primero la sesión de la cámara para evitar que el primer mensaje se
pierda.

Los focos y el modo de imagen tienen controles separados. El firmware de esta
cámara no permite garantizar focos blancos encendidos mientras permanece en
modo diurno.

## Reconocimiento facial experimental

La galería incluida usa una subcarpeta por identidad. Para ejecutar el panel
directamente con la galería versionada:

```powershell
.\.venv\Scripts\python control_panel.py --credentials-file .\claves.txt `
  --host 192.168.10.143 `
  --faces-dir .\face_gallery
```

YuNet detecta y alinea el rostro; SFace genera el embedding y lo compara por
similitud coseno contra una plantilla promedio por persona. El umbral inicial
es `0.50` y puede ajustarse con `--face-threshold`. La función es experimental:
un resultado facial se muestra en pantalla y selecciona el texto del aviso, pero
no debe usarse como única evidencia de identidad. La detección de persona genera
el aviso de entrada y el reconocimiento posterior agrega la bienvenida con el
nombre. La secuencia conserva la misma ventana de rearme de 10–15 segundos.

Ultralytics se distribuye bajo AGPL-3.0 y ofrece una licencia Enterprise. Debe
revisarse su licencia antes de comercializar o desplegar este prototipo como
software cerrado.
