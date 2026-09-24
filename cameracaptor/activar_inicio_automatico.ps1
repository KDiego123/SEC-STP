$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonw = Join-Path $projectDir ".venv\Scripts\pythonw.exe"
$panel = Join-Path $projectDir "control_panel.py"
$credentials = Join-Path $projectDir "claves.txt"
$faces = Join-Path $projectDir "face_gallery"

if (-not (Test-Path -LiteralPath $pythonw)) {
    throw "No existe el entorno virtual. Ejecuta instalar.bat primero."
}

$startupDir = [Environment]::GetFolderPath("Startup")
$shortcutPath = Join-Path $startupDir "CameraCaptor Panel.lnk"
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $pythonw
$shortcut.Arguments = ('"{0}" --credentials-file "{1}" --host "192.168.1.51" --faces-dir "{2}"' -f $panel, $credentials, $faces)
$shortcut.WorkingDirectory = $projectDir
$shortcut.WindowStyle = 1
$shortcut.Description = "Panel y API local de CameraCaptor"
$shortcut.Save()

Write-Host "Inicio automático activado para este usuario."
Write-Host "Acceso directo: $shortcutPath"
Write-Host "El panel se abrirá en el próximo inicio de sesión de Windows."
