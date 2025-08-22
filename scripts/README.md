# Scripts Directory

This directory contains utility scripts for setting up and managing the Medical Superbill PDF Extraction System.

## Available Scripts

### `setup_azure_env.ps1`
PowerShell script to configure Azure OpenAI environment variables.

**Usage:**
```powershell
# Navigate to project root
cd "d:\Repo\PDF"

# Run the setup script
.\scripts\setup_azure_env.ps1
```

**What it does:**
- Sets Azure OpenAI environment variables for the current PowerShell session
- Provides instructions for permanent environment variable setup
- Displays helpful messages for next steps

**Required Configuration:**
Before running, edit the script to replace placeholder values:
- `your-resource-name` → Your actual Azure OpenAI resource name
- `your-api-key-here` → Your actual Azure OpenAI API key
- Deployment name if different from `gpt-4o`

## Best Practices

- Keep all setup and utility scripts in this directory
- Use descriptive names for scripts
- Include usage instructions in script comments
- Test scripts before committing to repository
