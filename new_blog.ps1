param (
    [Parameter(Mandatory=$true)]
    [string]$Title,

    [Parameter(Mandatory=$true)]
    [ValidateSet("tech", "note", "write", "disclaimer")]
    [string]$Category
)

# 映射英文分类到中文类别
$CategoryMap = @{
    "tech" = "技术"
    "note" = "笔记"
    "write" = "随笔"
    "disclaimer" = "声明"
}

$ChineseCategory = $CategoryMap[$Category]

# 将标题转换为文件夹名字 (全小写，空格替换为连字符)
$Slug = $Title.ToLower() -replace '\s+', '-' -replace '[^\w\-]', ''

# 获取当前标准时间格式 (ISO 8601)
$CurrentDate = (Get-Date).ToString("yyyy-MM-ddTHH:mm:ssK")

# 确定路径
$FolderPath = "content\posts\$Category\$Slug"
$FilePath = "$FolderPath\index.md"

# 创建文件夹
if (-not (Test-Path $FolderPath)) {
    New-Item -ItemType Directory -Path $FolderPath | Out-Null
}

# 填充前台文件 (Front Matter)
$FrontMatter = @"
---
title: "$Title"
date: $CurrentDate
draft: false
categories: ["$ChineseCategory"]
tags: [""]
lightgallery: true
---


"@

# 写入文件
[System.IO.File]::WriteAllText((Resolve-Path .).Path + "\$FilePath", $FrontMatter, [System.Text.Encoding]::UTF8)

Write-Host "成功创建新博客！" -ForegroundColor Green
Write-Host "路径: $FilePath" -ForegroundColor Cyan
Write-Host "你可以直接打开这个文件开始写作了。"
