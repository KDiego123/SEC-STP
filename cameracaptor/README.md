# CameraCaptor

Panel local para controlar una cámara IP mediante RTSP, HTTP y ONVIF.

Incluye:

- visualización del vídeo RTSP;
- detección de personas con YOLO11 nano;
- aviso de voz por el altavoz de la cámara;
- escucha del micrófono de la cámara y órdenes de voz offline;
- control independiente de iluminación y visión nocturna;
- movimiento PTZ mediante botones o teclado.

## Instalación

En Windows, instala Python 3 (probado con Python 3.13) y ejecuta una sola vez:

```powershell
.\instalar.bat
```

El branch incluye el modelo YOLO, los modelos faciales YuNet/SFace, la galería
de identidades y `claves.txt`, tal como se requiere para esta cámara en la red local.
No hace falta instalar FFmpeg por separado: `requirements.txt` instala una copia
compatible mediante `imageio-ffmpeg`.

## Ejecución

```powershell
.\ejecutar_panel.bat
```

El script busca automáticamente la cámara mediante ONVIF en todas las interfaces
IPv4 activas y verifica su identidad con la MAC `E8:B7:23:47:95:27`. Esto permite
usar la conexión Ethernet directa (`192.168.1.51`) o una IP entregada por el
router sin editar archivos. La IP del lanzador solo funciona como respaldo.

Para indicar manualmente una IP preferida:

```powershell
.\ejecutar_panel.bat 192.168.1.51
```

El umbral de ausencia previo a un nuevo aviso de persona puede ajustarse entre
10 y 15 segundos:

```powershell
.\.venv\Scripts\python control_panel.py --credentials-file .\claves.txt `
  --host 192.168.1.51 --faces-dir .\face_gallery `
  --presence-seconds 1.5 --idle-seconds 12
```

## Funcionamiento

YOLO procesa solamente la clase `person`. El modelo se prepara al arrancar y
analiza hasta diez fotogramas por segundo. Después del intervalo configurado
sin personas, una detección clara (confianza de al menos `0.70`) se valida en
`0.6` segundos; una detección dudosa conserva la validación de `1.5` segundos
para reducir falsos positivos. Al cumplirse ese tiempo, el aviso se envía
inmediatamente y una detección negativa reinicia la validación. Estos valores
pueden ajustarse con `--fast-presence-seconds`, `--presence-seconds` y
`--strong-person-confidence`.
Primero dice «Se ha identificado a alguien en la entrada». La llegada permanece
activa y, si después reconoce un rostro conocido, interrumpe ese aviso general y
da prioridad inmediata a una frase como «Bienvenido Brayan». Si la identidad ya
está disponible desde el principio, omite el aviso general. Las frases no se
repiten continuamente.

El TTS se sintetiza con Windows SAPI y se transmite como audio G.711 μ-law por
el backchannel RTSP. El programa prepara el altavoz al arrancar y envía un pulso
silencioso cada 30 segundos para impedir que el firmware duerma el canal de
audio. Si ese mantenimiento falla, conserva el precalentamiento anterior como
respaldo antes del siguiente aviso. Los avisos de entrada y las bienvenidas
conocidas se sintetizan en segundo plano al abrir el panel. Si aparece una
identidad durante el aviso general, la bienvenida sustituye el audio activo sin
repetir el precalentamiento.

Los focos y el modo de imagen tienen controles separados. El firmware de esta
cámara no permite garantizar focos blancos encendidos mientras permanece en
modo diurno.

## Micrófono y órdenes de voz

La pestaña **Voz** permite escuchar en el equipo el micrófono integrado de la
cámara. En una sesión de Escritorio remoto, Windows debe estar configurado para
reproducir el audio remoto en el equipo desde el que se realiza la conexión.

El reconocimiento funciona localmente, sin enviar grabaciones a Internet. Con
la opción de órdenes activada acepta estas frases completas:

- «Prende las luces»: fuerza la visión nocturna y enciende los focos blancos.
- «Apaga las luces»: apaga los focos blancos y conserva la visión nocturna con
  iluminación infrarroja.

Mientras la cámara reproduce un aviso TTS, la escucha local se silencia y las
órdenes se ignoran brevemente para evitar que la cámara obedezca su propia voz.
Se incluye el modelo `vosk-model-small-es-0.42`, publicado por Vosk bajo
Apache-2.0.

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
