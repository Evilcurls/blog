# Blog Migration Script v2 - reads mapping from JSON to handle Chinese filenames
$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$srcRoot = "d:\gitcode\blog\docs"
$dstRoot = "d:\gitcode\blog-hugo\content\posts"
$blogGitDir = "d:\gitcode\blog"
$mapFile = "d:\gitcode\blog-hugo\posts_map.json"

# Clean existing posts
if (Test-Path $dstRoot) { Remove-Item -Recurse -Force $dstRoot }
New-Item -Path $dstRoot -ItemType Directory -Force | Out-Null

# Read mapping
$jsonContent = [System.IO.File]::ReadAllText($mapFile, [System.Text.Encoding]::UTF8)
$posts = $jsonContent | ConvertFrom-Json

$successCount = 0
$failCount = 0

foreach ($post in $posts) {
    $srcPath = Join-Path $srcRoot $post.src
    $slug = $post.slug
    $dstDir = Join-Path $dstRoot $slug
    $dstFile = Join-Path $dstDir "index.md"

    Write-Host "[$($successCount+$failCount+1)/$($posts.Count)] $($post.src) -> $slug" -ForegroundColor Cyan

    if (-not (Test-Path $srcPath)) {
        Write-Host "  SKIP: Not found: $srcPath" -ForegroundColor Yellow
        $failCount++
        continue
    }

    # Read source
    $content = [System.IO.File]::ReadAllText($srcPath, [System.Text.Encoding]::UTF8)
    $lines = $content -split "`n"

    # Extract title
    $title = ""
    $headIdx = -1
    $headLevel = 0
    for ($i = 0; $i -lt $lines.Count; $i++) {
        $l = $lines[$i].TrimEnd("`r")
        if ($l -match "^(#{1,6})\s+(.+)$") {
            $headLevel = $Matches[1].Length
            $title = $Matches[2].Trim()
            $headIdx = $i
            break
        }
    }
    if ([string]::IsNullOrEmpty($title)) {
        $title = [System.IO.Path]::GetFileNameWithoutExtension(($post.src -split "\\")[-1])
    }

    # Get date from git
    $relGitPath = "docs/" + ($post.src -replace "\\", "/")
    $dateStr = ""
    try {
        $gitOut = & git -C $blogGitDir log --diff-filter=A --follow "--format=%aI" -- $relGitPath 2>$null
        if ($gitOut) {
            if ($gitOut -is [array]) { $dateStr = $gitOut[-1] } else { $dateStr = $gitOut }
        }
    } catch {}
    if ([string]::IsNullOrEmpty($dateStr)) {
        try {
            $gitOut = & git -C $blogGitDir log --follow "--format=%aI" -- $relGitPath 2>$null
            if ($gitOut) {
                if ($gitOut -is [array]) { $dateStr = $gitOut[-1] } else { $dateStr = $gitOut }
            }
        } catch {}
    }
    if ([string]::IsNullOrEmpty($dateStr)) {
        $fi = Get-Item $srcPath
        $dateStr = $fi.LastWriteTime.ToString("yyyy-MM-ddTHH:mm:sszzz")
    }

    # Build front matter
    $draft = if ($post.draft) { "true" } else { "false" }
    $tagsStr = ($post.tags | ForEach-Object { "`"$_`"" }) -join ", "
    $escapedTitle = $title -replace '"', '\"'

    $sb = [System.Text.StringBuilder]::new()
    [void]$sb.AppendLine("---")
    [void]$sb.AppendLine("title: `"$escapedTitle`"")
    [void]$sb.AppendLine("date: $dateStr")
    [void]$sb.AppendLine("draft: $draft")
    [void]$sb.AppendLine("categories: [`"$($post.cat)`"]")
    [void]$sb.AppendLine("tags: [$tagsStr]")
    [void]$sb.AppendLine("math: true")
    [void]$sb.AppendLine("lightgallery: true")
    [void]$sb.AppendLine("---")
    [void]$sb.AppendLine("")

    # Body: skip h1 heading line
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($i -eq $headIdx -and $headLevel -eq 1) { continue }
        [void]$sb.Append($lines[$i])
        if ($i -lt $lines.Count - 1) { [void]$sb.Append("`n") }
    }

    # Write file
    New-Item -Path $dstDir -ItemType Directory -Force | Out-Null
    [System.IO.File]::WriteAllText($dstFile, $sb.ToString(), [System.Text.UTF8Encoding]::new($false))

    Write-Host "  Title: $title | Date: $dateStr" -ForegroundColor Green

    # Copy images from source directory
    $srcDir = Split-Path $srcPath -Parent
    $exts = @("*.png","*.jpg","*.jpeg","*.gif","*.svg","*.webp","*.mp3","*.mp4")
    foreach ($ext in $exts) {
        Get-ChildItem -Path $srcDir -Filter $ext -File -ErrorAction SilentlyContinue | ForEach-Object {
            Copy-Item $_.FullName -Destination $dstDir -Force
            Write-Host "  img: $($_.Name)" -ForegroundColor DarkGray
        }
    }

    # Copy res/ and img/ subdirectories
    foreach ($subDirName in @("res","img")) {
        $sd = Join-Path $srcDir $subDirName
        if (Test-Path $sd) {
            $dd = Join-Path $dstDir $subDirName
            New-Item -Path $dd -ItemType Directory -Force | Out-Null
            Copy-Item -Path "$sd\*" -Destination $dd -Recurse -Force
            Write-Host "  dir: $subDirName/" -ForegroundColor DarkGray
        }
    }

    $successCount++
}

Write-Host ""
Write-Host "===== Migration Done =====" -ForegroundColor White
Write-Host "Success: $successCount / $($posts.Count)" -ForegroundColor Green
if ($failCount -gt 0) { Write-Host "Failed:  $failCount" -ForegroundColor Red }
