# Loopback-only development server. No remote model calls or shell execution.
$ErrorActionPreference = 'Stop'
$reflexRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $reflexRoot
& "$reflexRoot\.venv\Scripts\python.exe" -m agent_reflex.cli serve --factory examples.server_app:build_engine --host 127.0.0.1 --port 8080 --allow-unauthenticated
exit $LASTEXITCODE
