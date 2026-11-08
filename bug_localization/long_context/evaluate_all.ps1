[CmdletBinding()]
param(
    [string]$DatasetFile,
    [string]$Config = "$PSScriptRoot/settings.example.json",
    [string]$OutputDir = "$PSScriptRoot/outputs/evaluation50",
    [string]$CredentialsFile,
    [switch]$PrepareOnly
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$python = Join-Path $repositoryRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'The repository .venv is missing.' }
$Config = (Resolve-Path -LiteralPath $Config).Path

# Read only the named credential, without executing a dotenv file or printing it.
if ($CredentialsFile) {
    $credentialText = [IO.File]::ReadAllText((Resolve-Path -LiteralPath $CredentialsFile).Path)
    $match = [regex]::Match($credentialText, '(?m)^\s*(?:export\s+)?OPENAI_API_KEY\s*=\s*(.*?)\s*$')
    if (-not $match.Success) { throw 'Credentials file has no OPENAI_API_KEY assignment.' }
    $env:OPENAI_API_KEY = $match.Groups[1].Value.Trim().Trim('"').Trim("'")
    $credentialText = $null
    $match = $null
}
if (-not $env:OPENAI_API_KEY) {
    foreach ($scope in @('User', 'Machine')) {
        $storedKey = [Environment]::GetEnvironmentVariable('OPENAI_API_KEY', $scope)
        if ($storedKey) { $env:OPENAI_API_KEY = $storedKey; break }
    }
}
if (-not $PrepareOnly -and [string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY)) {
    throw 'Set OPENAI_API_KEY or pass -CredentialsFile before a live run. No model requests were made.'
}

if (-not $DatasetFile) {
    $cacheRoot = if ($env:HF_DATASETS_CACHE) { $env:HF_DATASETS_CACHE } else {
        Join-Path ([Environment]::GetFolderPath('UserProfile')) '.cache/huggingface/datasets'
    }
    $datasetRoot = Join-Path $cacheRoot 'JetBrains-Research___lca-bug-localization/py'
    if (Test-Path -LiteralPath $datasetRoot) {
        $cached = Get-ChildItem -LiteralPath $datasetRoot -Recurse -Filter 'lca-bug-localization-test.arrow' |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($cached) { $DatasetFile = $cached.FullName }
    }
}
if ($DatasetFile) { $DatasetFile = (Resolve-Path -LiteralPath $DatasetFile).Path }

Push-Location -LiteralPath $repositoryRoot
try {
    if ($DatasetFile) {
        $verificationCode = "import sys; from datasets import Dataset; d=Dataset.from_file(sys.argv[1]); assert len(d)==50, 'Expected exactly 50 test examples'; print('Verified 50 cached test examples; fingerprint='+d._fingerprint)"
        & $python -c $verificationCode $DatasetFile
        if ($LASTEXITCODE -ne 0) { throw 'Dataset verification failed.' }
    }
    $arms = @(
        @{ Name = 'direct_small'; Mode = 'direct'; Budget = 8192 },
        @{ Name = 'rag_small'; Mode = 'rag'; Budget = 8192 },
        @{ Name = 'direct_large'; Mode = 'direct'; Budget = 65536 },
        @{ Name = 'rag_large'; Mode = 'rag'; Budget = 65536 },
        @{ Name = 'issue_only'; Mode = 'issue_only'; Budget = 0 }
    )
    $runRoots = @()
    foreach ($arm in $arms) {
        $armRoot = if ($PrepareOnly) { Join-Path $OutputDir "preparation/$($arm.Name)" } else {
            Join-Path $OutputDir $arm.Name
        }
        $runRoots += $armRoot
        $runArgs = @('-m', 'bug_localization.long_context.run', '--mode', $arm.Mode,
            '--repo-token-budget', [string]$arm.Budget, '--config', $Config,
            '--limit', '50', '--output-dir', $armRoot)
        if ($DatasetFile) { $runArgs += @('--dataset-file', $DatasetFile) }
        if ($PrepareOnly) { $runArgs += '--dry-run' }
        & $python @runArgs
        # Nonzero can denote recorded fallbacks; continue to report every arm.
        if (-not (Test-Path -LiteralPath (Join-Path $armRoot 'latest_run.json'))) {
            throw "No run artifacts produced for $($arm.Name)."
        }
    }
    if ($PrepareOnly) {
        Write-Output "Prepared prompts only; no model calls or benchmark scores. Outputs: $OutputDir/preparation"
    } else {
        $compareArgs = @('-m', 'bug_localization.long_context.compare') + $runRoots +
            @('--output-dir', (Join-Path $OutputDir 'comparison'))
        if ($DatasetFile) { $compareArgs += @('--dataset-file', $DatasetFile) }
        & $python @compareArgs
        if ($LASTEXITCODE -ne 0) { throw 'Comparison failed; inspect the saved run summaries.' }
    }
} finally {
    Pop-Location
}
