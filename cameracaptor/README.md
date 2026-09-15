# CameraCaptor

Prototipo local para captura RTSP de una cámara IP y posterior análisis visual.

## Vista previa

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python preview_camera.py --credentials-file .\claves.txt
```

Pulsa `q` o `Ctrl+C` para cerrar. La URL completa y las credenciales no se imprimen.

La integración YOLO queda deliberadamente pendiente hasta confirmar que esta captura es estable.

## Demo interactiva

```powershell
.\.venv\Scripts\python demo_camera.py --credentials-file .\claves.txt
```

Controles: `Q` salir, `S` guardar captura, `M` activar/desactivar movimiento y
`T` activar/desactivar las alertas de voz locales de Windows. La voz comienza
desactivada. Los eventos se escriben localmente en `events/events.jsonl`.

## YOLO: detección de personas

```powershell
.\.venv\Scripts\python yolo_camera.py --credentials-file .\claves.txt
```

Pulsa `Q` para salir o `S` para guardar una captura. Los parámetros
`--confidence`, `--imgsz`, `--every`, `--device` y `--model` permiten ajustar
la inferencia. Por defecto solo se procesa la clase `person`.

Ultralytics se distribuye bajo AGPL-3.0 y ofrece licencia Enterprise. La empresa
debe revisar la licencia antes de comercializar o desplegar este prototipo como
software cerrado.

## Panel de YOLO, voz, focos y PTZ

```powershell
.\.venv\Scripts\python control_panel.py --credentials-file .\claves.txt
```

YOLO11 nano detecta únicamente personas y dibuja sus cajas sobre el vídeo.
La casilla del panel permite pausarlo. Tras 12 segundos observados sin personas,
dos inferencias positivas consecutivas disparan una sola vez el TTS
"Se ha identificado un humano" por el altavoz de la cámara. No se repite
mientras alguien siga presente; otra alerta exige una nueva ausencia de 12 s.
El umbral puede elegirse entre 10 y 15 s con `--idle-seconds 12`. Si el vídeo
se desconecta o se pausa YOLO, el tiempo de inactividad se reinicia. El modelo
`yolo11n.pt` se descarga automáticamente en el primer arranque si no está
presente; las ejecuciones posteriores reutilizan la copia local.

Escribe hasta 200 caracteres y pulsa **Hablar**. El PC sintetiza la voz con
Windows SAPI y transmite audio G.711 μ-law al altavoz de la cámara por el
backchannel RTSP. No se guarda audio. Las flechas PTZ funcionan por mouse o
teclado; suelta el control para detener el movimiento. Cada orden tiene además
un timeout ONVIF de 0,4 segundos y velocidad limitada. Este prototipo utiliza
la interfaz HTTP/RTSP local de la cámara: no expongas sus puertos a Internet.

Si el altavoz lleva 45 segundos o más sin usarse, se abre primero una sesión
RTSP silenciosa de 0,8 s para reactivar el canal y después se transmite la voz.
El panel confirma el resultado del envío; si falla el aviso automático de una
persona, lo reintenta una sola vez tras 2 s mientras la detección siga vigente.

El panel tiene controles separados. **Focos blancos** cambia solamente la
selección de lámpara (blanca, IR o modo original). **Visión nocturna** cambia
solamente el modo de imagen (noche, día o modo original/automático). En la
prueba física, elegir IR con noche forzada apagó los focos blancos y dejó la
imagen gris. Los cambios persisten en la cámara hasta elegir otro modo.

Los controles laterales se organizan en las pestañas **Voz**, **Iluminación**
y **PTZ** para que todos sean accesibles en pantallas de 1024×768. El estado
RTSP queda en el encabezado, el estado de YOLO bajo el vídeo y los mensajes de
operación en la barra inferior.

Hay una limitación del firmware: los focos blancos no se encienden físicamente
en modo diurno aunque estén seleccionados. Activar visión nocturna mientras la
lámpara blanca está seleccionada puede encenderlos; desactivar visión nocturna
los apaga. No se puede garantizar la combinación «focos blancos encendidos +
visión diurna» con los controles ONVIF/web que la cámara anuncia. El panel
indica cuando la lámpara blanca queda seleccionada esperando el modo noche.
