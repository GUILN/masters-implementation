# Secrets Configuration

This project uses a `secrets.ini` file to store sensitive information like API keys and DSNs.

## Setup

1. **Copy the template:**
   ```bash
   cp secrets.ini.template secrets.ini
   ```

2. **Edit the secrets file:**
   ```bash
   nano secrets.ini  # or your preferred editor
   ```

3. **Add your Sentry DSN:**
   - Go to your Sentry project settings
   - Copy the DSN from the "Client Keys (DSN)" section
   - Replace `https://your-sentry-dsn@sentry.io/project-id` with your actual DSN

## Example secrets.ini

```ini
[sentry]
dsn = https://abc123@o123456.ingest.sentry.io/123456
environment = production

[api_keys]
# Add other API keys as needed
# openai_api_key = sk-your-key
```

## Security Notes

- The `secrets.ini` file is automatically ignored by git (see `.gitignore`)
- Never commit sensitive information to version control
- Use different DSNs for different environments (development, staging, production)
- The `secrets.ini.template` file shows the structure but contains no real secrets

## Usage in Code

The secrets are automatically loaded by the `ExtractionConfig` class:

```python
config = ExtractionConfig()
sentry_settings = config.sentry_settings
print(sentry_settings['dsn'])  # Your Sentry DSN
```
