# SPDX-License-Identifier: MPL-2.0
# Capture one child process without a shell or visible window. Memory maxima are sampled,
# not true peaks; NVIDIA readings cover the whole device, not this process's allocation.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ExecutablePath,
    [string[]]$Arguments = @(),
    [Parameter(Mandatory = $true)][string]$EvidenceDirectory,
    [Parameter(Mandatory = $true)][string]$Device,
    [string]$FixturePath,
    [string]$NvidiaSmiPath,
    [string]$WorkingDirectory = (Get-Location).ProviderPath,
    [ValidateRange(50, 10000)][int]$SampleIntervalMs = 250,
    [ValidateRange(1, 86400)][int]$TimeoutSeconds = 900
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Resolve-InputFile([string]$PathValue) {
    $resolved = [System.IO.Path]::GetFullPath($PathValue)
    if (-not [System.IO.File]::Exists($resolved)) { throw "Input file does not exist: $resolved" }
    return $resolved
}

function Get-GpuSnapshot([string]$ToolPath) {
    if (-not $ToolPath) { return $null }
    $query = [System.Diagnostics.Process]::new()
    $query.StartInfo.FileName = $ToolPath
    $query.StartInfo.UseShellExecute = $false
    $query.StartInfo.CreateNoWindow = $true
    $query.StartInfo.RedirectStandardOutput = $true
    $query.StartInfo.RedirectStandardError = $true
    $query.StartInfo.ArgumentList.Add('--query-gpu=index,name,memory.used,memory.total,utilization.gpu')
    $query.StartInfo.ArgumentList.Add('--format=csv,noheader,nounits')
    try {
        [void]$query.Start()
        $output = $query.StandardOutput.ReadToEndAsync()
        $errors = $query.StandardError.ReadToEndAsync()
        if (-not $query.WaitForExit(5000)) {
            $query.Kill($true)
            $query.WaitForExit()
            return [ordered]@{ status = 'query-timeout'; rows = @() }
        }
        $stdoutText = $output.GetAwaiter().GetResult()
        $stderrText = $errors.GetAwaiter().GetResult()
        if ($query.ExitCode -ne 0) {
            return [ordered]@{ status = 'query-failed'; exit_code = $query.ExitCode; stderr = $stderrText; rows = @() }
        }
        $rows = @($stdoutText | ConvertFrom-Csv -Header index, name, memory_used_mib, memory_total_mib, utilization_gpu_percent | ForEach-Object {
            [ordered]@{
                index = $_.index.Trim()
                name = $_.name.Trim()
                memory_used_mib = $_.memory_used_mib.Trim()
                memory_total_mib = $_.memory_total_mib.Trim()
                utilization_gpu_percent = $_.utilization_gpu_percent.Trim()
            }
        })
        return [ordered]@{ status = 'sampled'; rows = $rows }
    } catch {
        return [ordered]@{ status = 'query-error'; error = $_.Exception.Message; rows = @() }
    } finally {
        $query.Dispose()
    }
}

$executable = Resolve-InputFile $ExecutablePath
$fixture = if ($FixturePath) { Resolve-InputFile $FixturePath } else { $null }
$gpuTool = if ($NvidiaSmiPath) { Resolve-InputFile $NvidiaSmiPath } else { $null }
$workingRoot = [System.IO.Path]::GetFullPath($WorkingDirectory)
if (-not [System.IO.Directory]::Exists($workingRoot)) { throw 'Working directory does not exist.' }
if ($Arguments.Count -gt 256) { throw 'At most 256 child arguments are accepted.' }
$argumentBytes = 0
foreach ($argument in $Arguments) {
    if ($null -eq $argument -or $argument.Length -gt 4096 -or $argument -match '[^\x20-\x7e]') {
        throw 'Child arguments must be printable ASCII strings of at most 4096 characters.'
    }
    $argumentBytes += $argument.Length
}
if ($argumentBytes -gt 65536) { throw 'Total child argument length exceeds 65536 ASCII bytes.' }
if ($Device.Length -gt 128 -or $Device -match '[^\x20-\x7e]') { throw 'Device label must be bounded printable ASCII.' }

$evidenceRoot = [System.IO.Path]::GetFullPath($EvidenceDirectory)
if (Test-Path -LiteralPath $evidenceRoot) { throw 'Evidence directory already exists; captures never overwrite prior evidence.' }
$executableDigest = (Get-FileHash -LiteralPath $executable -Algorithm SHA256).Hash.ToLowerInvariant()
$fixtureDigest = if ($fixture) { (Get-FileHash -LiteralPath $fixture -Algorithm SHA256).Hash.ToLowerInvariant() } else { $null }
[void][System.IO.Directory]::CreateDirectory($evidenceRoot)
$reportPath = Join-Path $evidenceRoot 'report.json'
$stderrPath = Join-Path $evidenceRoot 'stderr.txt'
$receiptPath = Join-Path $evidenceRoot 'receipt.json'
$stdoutFile = $null
$stderrFile = $null
$receiptFile = $null
$child = [System.Diagnostics.Process]::new()
$child.StartInfo.FileName = $executable
$child.StartInfo.WorkingDirectory = $workingRoot
$child.StartInfo.UseShellExecute = $false
$child.StartInfo.CreateNoWindow = $true
$child.StartInfo.RedirectStandardOutput = $true
$child.StartInfo.RedirectStandardError = $true
foreach ($argument in $Arguments) { $child.StartInfo.ArgumentList.Add($argument) }
$started = $false
try {
    # CreateNew also rejects a file appearing after the directory check. No existing bytes change.
    $stdoutFile = [System.IO.File]::Open($reportPath, 'CreateNew', 'Write', 'Read')
    $stderrFile = [System.IO.File]::Open($stderrPath, 'CreateNew', 'Write', 'Read')
    $receiptFile = [System.IO.File]::Open($receiptPath, 'CreateNew', 'Write', 'Read')
    $gpuBefore = Get-GpuSnapshot $gpuTool
    $startedAt = [DateTimeOffset]::UtcNow
    $clock = [System.Diagnostics.Stopwatch]::StartNew()
    [void]$child.Start()
    $started = $true
    $stdoutCopy = $child.StandardOutput.BaseStream.CopyToAsync($stdoutFile)
    $stderrCopy = $child.StandardError.BaseStream.CopyToAsync($stderrFile)
    $samples = [System.Collections.Generic.List[object]]::new()
    $maxWorking = [long]0
    $maxPrivate = [long]0
    $timedOut = $false
    while (-not $child.HasExited) {
        # Observe stream failures during execution instead of keeping the child running after a
        # failed capture (including disk-full errors). We never delete or clean up evidence.
        if ($stdoutCopy.IsFaulted) { [void]$stdoutCopy.GetAwaiter().GetResult() }
        if ($stderrCopy.IsFaulted) { [void]$stderrCopy.GetAwaiter().GetResult() }
        if ($clock.Elapsed.TotalSeconds -ge $TimeoutSeconds) {
            $timedOut = $true
            $child.Kill($true)
            break
        }
        $child.Refresh()
        if ($child.HasExited) { break }
        $working = $child.WorkingSet64
        $private = $child.PrivateMemorySize64
        $maxWorking = [Math]::Max($maxWorking, $working)
        $maxPrivate = [Math]::Max($maxPrivate, $private)
        $sampleAt = $clock.Elapsed.TotalSeconds
        $samples.Add([ordered]@{
            elapsed_seconds = $sampleAt
            working_set_bytes = $working
            private_bytes = $private
            gpu = (Get-GpuSnapshot $gpuTool)
        })
        Start-Sleep -Milliseconds $SampleIntervalMs
    }
    $child.WaitForExit()
    $elapsed = $clock.Elapsed.TotalSeconds
    [void]$stdoutCopy.GetAwaiter().GetResult()
    [void]$stderrCopy.GetAwaiter().GetResult()
    $stdoutFile.Flush($true)
    $stderrFile.Flush($true)
    $gpuAfter = Get-GpuSnapshot $gpuTool
    $exitCode = $child.ExitCode
    $receipt = [ordered]@{
        version = 1
        device = $Device
        exit_code = $exitCode
        started_at_utc = $startedAt.ToString('O')
        elapsed_seconds = $elapsed
        executable_sha256 = $executableDigest
        fixture_sha256 = $fixtureDigest
        executable = $executable
        arguments = $Arguments
        working_directory = $workingRoot
        timeout_seconds = $TimeoutSeconds
        timed_out = $timedOut
        memory_measurement = "$SampleIntervalMs ms requested process working/private sampling; sample maxima are not a measured peak"
        sample_interval_ms = $SampleIntervalMs
        memory_samples = $samples
        max_sampled_working_set_bytes = $maxWorking
        max_sampled_private_bytes = $maxPrivate
        gpu_memory_measurement = 'Whole-device NVIDIA memory snapshots; not attributable process allocations or true inference peak. Sampling includes monitor overhead.'
        gpu_before = $gpuBefore
        gpu_after = $gpuAfter
        report = $reportPath
        report_bytes = $stdoutFile.Length
        stderr = $stderrPath
        stderr_bytes = $stderrFile.Length
        acceptance_policy_changed = $false
    }
    $json = $receipt | ConvertTo-Json -Depth 10
    $bytes = [System.Text.UTF8Encoding]::new($false).GetBytes($json)
    $receiptFile.Write($bytes, 0, $bytes.Length)
    $receiptFile.Flush($true)
    Write-Output $json
    if ($timedOut) { throw "Child exceeded the $TimeoutSeconds second capture limit; evidence retained." }
    if ($exitCode -ne 0) { throw "Child failed with exit code $exitCode; evidence retained." }
} catch {
    if ($started -and -not $child.HasExited) {
        # Only this wrapper's child is stopped. Keep all partial evidence for operator review.
        $child.Kill($true)
        $child.WaitForExit()
    }
    throw
} finally {
    if ($null -ne $stdoutFile) { $stdoutFile.Dispose() }
    if ($null -ne $stderrFile) { $stderrFile.Dispose() }
    if ($null -ne $receiptFile) { $receiptFile.Dispose() }
    $child.Dispose()
}
