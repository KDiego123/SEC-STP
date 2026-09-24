$ErrorActionPreference = "Stop"

$startupDir = [Environment]::GetFolderPath("Startup")
$shortcutPath = Join-Path $startupDir "CameraCaptor Panel.lnk"

if (Test-Path -LiteralPath $shortcutPath) {
    Remove-Item -LiteralPath $shortcutPath -Force
    Write-Host "Inicio automático de CameraCaptor desactivado."
} else {
    Write-Host "CameraCaptor no estaba registrado en el inicio automático."
}
