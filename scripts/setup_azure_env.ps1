# PowerShell script to set Azure OpenAI environment variables
# Replace the placeholder values with your actual Azure OpenAI credentials

Write-Host "🔧 Azure OpenAI Environment Setup" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan

# Option 1: Set for current PowerShell session only
Write-Host "`n📝 Setting environment variables for current session..." -ForegroundColor Yellow

$env:AZURE_OPENAI_ENDPOINT = "https://your-resource-name.openai.azure.com/"
$env:AZURE_OPENAI_API_KEY = "your-api-key-here"
$env:AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4o"  # or your deployed model name
$env:AZURE_OPENAI_API_VERSION = "2024-02-01"

# Option 2: Set permanently for your user (uncomment lines below)
Write-Host "`n💾 To set permanently, uncomment these lines:" -ForegroundColor Green
Write-Host "# [Environment]::SetEnvironmentVariable('AZURE_OPENAI_ENDPOINT', 'https://your-resource-name.openai.azure.com/', 'User')" -ForegroundColor Gray
Write-Host "# [Environment]::SetEnvironmentVariable('AZURE_OPENAI_API_KEY', 'your-api-key-here', 'User')" -ForegroundColor Gray
Write-Host "# [Environment]::SetEnvironmentVariable('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4o', 'User')" -ForegroundColor Gray
Write-Host "# [Environment]::SetEnvironmentVariable('AZURE_OPENAI_API_VERSION', '2024-02-01', 'User')" -ForegroundColor Gray

Write-Host "`n✅ Azure OpenAI environment variables configured for current session." -ForegroundColor Green
Write-Host "⚠️  Remember to replace placeholder values with actual credentials!" -ForegroundColor Yellow
Write-Host "`n🧪 Test your configuration by running:" -ForegroundColor Cyan
Write-Host "python -m pytest tests/test_azure_connection.py -v" -ForegroundColor White
