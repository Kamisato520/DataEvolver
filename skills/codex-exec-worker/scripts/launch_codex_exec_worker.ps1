param(
    [string]$PromptFile,
    [string]$Workspace = (Get-Location).Path,
    [string]$RunRoot = "",
    [string]$Label = "codex-exec-worker",
    [string]$Model = "",
    [ValidateSet("read-only", "workspace-write", "danger-full-access")]
    [string]$Sandbox = "workspace-write",
    [string[]]$AddDir = @(),
    [switch]$Json,
    [switch]$Wait,
    [switch]$NoLaunch,
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = "Stop"

function Resolve-ExistingPath {
    param([Parameter(Mandatory = $true)][string]$PathValue)
    return (Resolve-Path -LiteralPath $PathValue).Path
}

function Normalize-Label {
    param([string]$Text)
    if ([string]::IsNullOrWhiteSpace($Text)) {
        $value = "codex-exec-worker"
    }
    else {
        $value = $Text.ToLowerInvariant()
    }
    $value = [System.Text.RegularExpressions.Regex]::Replace($value, "[^a-z0-9\-]+", "-")
    $value = [System.Text.RegularExpressions.Regex]::Replace($value, "-{2,}", "-")
    $value = $value.Trim("-")
    if ([string]::IsNullOrWhiteSpace($value)) {
        return "codex-exec-worker"
    }
    return $value
}

if (-not (Get-Command codex -ErrorAction SilentlyContinue)) {
    throw "Cannot find `codex` in PATH."
}

if ([string]::IsNullOrWhiteSpace($PromptFile)) {
    throw "-PromptFile is required."
}

$workspacePath = Resolve-ExistingPath -PathValue $Workspace
$promptSourcePath = Resolve-ExistingPath -PathValue $PromptFile

if ([string]::IsNullOrWhiteSpace($RunRoot)) {
    $RunRoot = Join-Path $workspacePath ".codex-exec-runs"
}

$runRootPath = [System.IO.Path]::GetFullPath($RunRoot)
New-Item -ItemType Directory -Force -Path $runRootPath | Out-Null

$safeLabel = Normalize-Label -Text $Label
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runDir = Join-Path $runRootPath "${timestamp}_${safeLabel}"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$promptTargetPath = Join-Path $runDir "prompt.txt"
$manifestPath = Join-Path $runDir "run_manifest.json"
$runnerPath = Join-Path $runDir "worker-run.ps1"
$lastMessagePath = Join-Path $runDir "last_message.txt"
$exitStatusPath = Join-Path $runDir "exit_status.json"
$streamPath = Join-Path $runDir ($(if ($Json) { "events.jsonl" } else { "stream.log" }))

Copy-Item -LiteralPath $promptSourcePath -Destination $promptTargetPath -Force

$resolvedAddDirs = @()
foreach ($dir in $AddDir) {
    $resolvedAddDirs += Resolve-ExistingPath -PathValue $dir
}

$codexArgs = @(
    "exec",
    "--sandbox", $Sandbox,
    "-C", $workspacePath,
    "-o", $lastMessagePath
)

if ($Json) {
    $codexArgs += "--json"
}

if (-not [string]::IsNullOrWhiteSpace($Model)) {
    $codexArgs += @("--model", $Model)
}

foreach ($dir in $resolvedAddDirs) {
    $codexArgs += @("--add-dir", $dir)
}

if ($ExtraArgs.Count -gt 0) {
    $codexArgs += $ExtraArgs
}

$codexArgs += "-"

$manifest = [ordered]@{
    created_at = (Get-Date).ToString("o")
    workspace = $workspacePath
    run_dir = $runDir
    prompt_path = $promptTargetPath
    last_message_path = $lastMessagePath
    stream_path = $streamPath
    exit_status_path = $exitStatusPath
    json_mode = [bool]$Json
    sandbox = $Sandbox
    model = $Model
    add_dirs = $resolvedAddDirs
    codex_args = $codexArgs
}

$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

$manifestPathLiteral = $manifestPath.Replace("'", "''")
$runnerContent = @"
`$ErrorActionPreference = 'Stop'
`$manifest = Get-Content -Raw -LiteralPath '$manifestPathLiteral' | ConvertFrom-Json
Set-Location -LiteralPath ([string]`$manifest.workspace)

`$argsList = @()
foreach (`$arg in `$manifest.codex_args) {
    `$argsList += [string]`$arg
}

try {
    Get-Content -Raw -LiteralPath ([string]`$manifest.prompt_path) |
        & codex @argsList 2>&1 |
        Tee-Object -FilePath ([string]`$manifest.stream_path)
    `$exitCode = `$LASTEXITCODE
}
catch {
    (`$_ | Out-String) | Tee-Object -FilePath ([string]`$manifest.stream_path) -Append | Out-Null
    `$exitCode = 1
}

`$status = [ordered]@{
    finished_at = (Get-Date).ToString('o')
    exit_code = `$exitCode
}

`$status | ConvertTo-Json | Set-Content -LiteralPath ([string]`$manifest.exit_status_path) -Encoding UTF8
exit `$exitCode
"@

Set-Content -LiteralPath $runnerPath -Value $runnerContent -Encoding UTF8

$process = $null
if (-not $NoLaunch) {
    $process = Start-Process `
        -FilePath "powershell" `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $runnerPath) `
        -WorkingDirectory $workspacePath `
        -PassThru
}

$manifest.launch = [ordered]@{
    prepared_at = (Get-Date).ToString("o")
    pid = $(if ($process) { $process.Id } else { $null })
    wait = [bool]$Wait
    no_launch = [bool]$NoLaunch
}
$manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

if ($Wait -and $process) {
    $process.WaitForExit()
}

[pscustomobject]@{
    run_dir = $runDir
    pid = $(if ($process) { $process.Id } else { $null })
    prompt_path = $promptTargetPath
    last_message_path = $lastMessagePath
    stream_path = $streamPath
    manifest_path = $manifestPath
    runner_path = $runnerPath
}
