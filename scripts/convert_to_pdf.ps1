param(
    [string]$DocxPath,
    [string]$PdfPath
)

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0

try {
    $doc = $word.Documents.Open($DocxPath)
    $doc.SaveAs([ref]$PdfPath, [ref]17)
    $doc.Close()
    Write-Output "Converted: $PdfPath"
} finally {
    $word.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
}
