# reports/book/nyc311_five_questions.html -> .pdf (Edge 헤드리스 인쇄)
# 실행: powershell -File reports/book/build_pdf.ps1
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$html = Join-Path $here "nyc311_five_questions.html"
$pdf  = Join-Path $here "nyc311_five_questions.pdf"
$edge = "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
$profile = Join-Path $env:TEMP "edge_pdf_profile"
$url = "file:///" + ($html -replace "\\", "/")
& $edge --headless=new --disable-gpu --no-pdf-header-footer --user-data-dir="$profile" --print-to-pdf="$pdf" $url 2>$null | Out-Null
Get-Item $pdf | Select-Object FullName, Length
