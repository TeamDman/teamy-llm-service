# Build the pinned Makepad adapter with actual CUDA, never its stub backend.
# Requires an existing CUDA toolkit and Visual Studio C++ build tools.
[CmdletBinding()]
param(
    [ValidatePattern('^[0-9]+[a-z]?$')][string]$CudaArchitecture = '89',
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$CargoArguments
)
$ErrorActionPreference = 'Stop'
if (-not $env:CUDA_PATH -or -not (Test-Path -LiteralPath (Join-Path $env:CUDA_PATH 'bin/nvcc.exe'))) {
    throw 'CUDA_PATH must point to an installed CUDA toolkit containing bin/nvcc.exe.'
}
if (-not $CargoArguments.Count) { throw 'Supply a Cargo command, for example build --locked -p teamy_llm_cli.' }
$previous = @{}
$settings = @{
    MAKEPAD_CUDA_ROOT = $env:CUDA_PATH
    MAKEPAD_GGML_REQUIRE_CUDA = '1'
    MAKEPAD_GGML_NO_CUDA = '0'
    MAKEPAD_GGML_CUDA_ARCH = $CudaArchitecture
}
try {
    foreach ($name in $settings.Keys) {
        $previous[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
        [Environment]::SetEnvironmentVariable($name, $settings[$name], 'Process')
    }
    & cargo @CargoArguments
    $cargoExit = $LASTEXITCODE
} finally {
    foreach ($name in $previous.Keys) {
        [Environment]::SetEnvironmentVariable($name, $previous[$name], 'Process')
    }
}
if ($cargoExit -ne 0) { throw "Cargo failed with exit code $cargoExit" }
