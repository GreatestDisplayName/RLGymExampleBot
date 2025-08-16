# This script acts as a simple wrapper to execute the Gemini CLI.
# It forwards all provided arguments to the 'gemini' command.

# Check if 'gemini' command is available in the system's PATH
if (-not (Get-Command -ErrorAction SilentlyContinue gemini)) {
    Write-Error "Error: 'gemini' command not found in your system's PATH."
    Write-Error "Please ensure the Gemini CLI is installed and accessible."
    exit 1
}

gemini $args
