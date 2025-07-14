# Manual Syntax Fixer for Schwabot Codebase
# This script manually fixes the most common E999 syntax errors

Write-Host "Manual Syntax Fixer - Schwabot Codebase" -ForegroundColor Green
Write-Host "=======================================" -ForegroundColor Green

$fixedCount = 0
$unicodeFixed = 0
$docstringFixed = 0

# Get all Python files
$pythonFiles = Get-ChildItem -Recurse -Filter "*.py" | Where-Object {
    $_.FullName -notmatch "\.git|__pycache__|\.venv|venv|node_modules"
}

Write-Host "Found $($pythonFiles.Count) Python files to process..." -ForegroundColor Yellow

foreach ($file in $pythonFiles) {
    try {
        $content = Get-Content $file.FullName -Raw -Encoding UTF8
        $originalContent = $content
        $fileFixed = $false
        
        # Fix stub docstring pattern
        if ($content -match '"""Stub main function\."""\."""') {
            $content = $content -replace '"""Stub main function\."""\."""', '"""Stub main function."""`n    pass`n'
            $docstringFixed++
            $fileFixed = $true
        }
        
        # Fix Unicode characters
        $unicodeReplacements = @{
            '∇' = 'del'
            '∈' = 'in'
            '≤' = '<='
            '≥' = '>='
            '⇒' = '=>'
            '∫' = 'int'
            '∂' = 'd'
            '·' = '.'
            '–' = '-'
            '₍' = '('
            '₎' = ')'
            '♦' = ''
            '×' = 'x'
            'Δ' = 'd'
            'Σ' = 'sum'
            'π' = 'pi'
            'σ' = 'sigma'
            'λ' = 'lambda'
            'μ' = 'mu'
            'α' = 'alpha'
            'β' = 'beta'
            'γ' = 'gamma'
            'δ' = 'delta'
            'ε' = 'epsilon'
            'θ' = 'theta'
            'φ' = 'phi'
            'ψ' = 'psi'
            'ω' = 'omega'
        }
        
        foreach ($unicode in $unicodeReplacements.Keys) {
            if ($content -match $unicode) {
                $content = $content -replace $unicode, $unicodeReplacements[$unicode]
                $unicodeFixed++
                $fileFixed = $true
            }
        }
        
        # Fix unterminated strings
        if ($content -match '"""([^"]*)\r?\n\s*"""\s*def\s+') {
            $content = $content -replace '"""([^"]*)\r?\n\s*"""\s*def\s+', '"""$1"""`n`ndef '
            $fileFixed = $true
        }
        
        if ($content -match '"""([^"]*)\r?\n\s*def\s+') {
            $content = $content -replace '"""([^"]*)\r?\n\s*def\s+', '"""$1"""`n`ndef '
            $fileFixed = $true
        }
        
        if ($content -match '"""([^"]*)\r?\n\s*if\s+__name__') {
            $content = $content -replace '"""([^"]*)\r?\n\s*if\s+__name__', '"""$1"""`n`nif __name__'
            $fileFixed = $true
        }
        
        # Write back if changes were made
        if ($fileFixed) {
            Set-Content -Path $file.FullName -Value $content -Encoding UTF8
            Write-Host "✅ Fixed: $($file.Name)" -ForegroundColor Green
            $fixedCount++
        }
        
    } catch {
        Write-Host "❌ Error processing $($file.Name): $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host "`nSummary:" -ForegroundColor Cyan
Write-Host "  Files processed: $($pythonFiles.Count)" -ForegroundColor White
Write-Host "  Files fixed: $fixedCount" -ForegroundColor Green
Write-Host "  Unicode fixes: $unicodeFixed" -ForegroundColor Yellow
Write-Host "  Docstring fixes: $docstringFixed" -ForegroundColor Yellow

Write-Host "`nManual syntax fixing completed!" -ForegroundColor Green 