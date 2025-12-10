# Environment Setup Guide

## Quick Setup (Minimal Configuration)

The **absolute minimum** you need to get started:

```bash
# 1. Copy example file
cp .env.example .env

# 2. Edit .env and add your OpenAI key
# Only this line is required:
OPENAI_API_KEY=sk-your-actual-key-here

# 3. Done! Run your first test
source venv/bin/activate
python examples/quick_test.py
```

## Environment Variables Reference

### Required Variables

Only **ONE** of these LLM API keys is required:

| Variable | Description | Where to Get | Cost |
|----------|-------------|--------------|------|
| `OPENAI_API_KEY` | OpenAI API key | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | ~$0.03/1K tokens |
| `ANTHROPIC_API_KEY` | Anthropic Claude key | [console.anthropic.com](https://console.anthropic.com/settings/keys) | ~$3/MTok |
| `DEEPSEEK_API_KEY` | DeepSeek API key | [platform.deepseek.com](https://platform.deepseek.com/) | ~$0.14/MTok |

### Optional Variables

All other variables in `.env.example` are **optional** and have sensible defaults:

#### LLM Configuration
- `LLM_PROVIDER` - Default: `openai`
- `STRATEGY_GENERATION_MODEL` - Default: `gpt-4`
- `CODE_GENERATION_MODEL` - Default: `gpt-4`
- `STRATEGY_TEMPERATURE` - Default: `0.7`
- `CODE_TEMPERATURE` - Default: `0.2`

#### Research Configuration
- `NUM_STRATEGIES_PER_BATCH` - Default: `10`
- `MAX_REFINEMENT_ATTEMPTS` - Default: `2`
- `STRATEGY_FAMILY_SIZE` - Default: `3`

#### Backtesting Configuration
- `INITIAL_CAPITAL` - Default: `10000`
- `SLIPPAGE_BPS` - Default: `5.0`

#### Filtering Criteria
- `MIN_SHARPE_RATIO` - Default: `0.5`
- `MIN_TOTAL_RETURN` - Default: `0.05` (5%)
- `MAX_DRAWDOWN` - Default: `0.30` (30%)

See `.env.example` for the complete list.

## Configuration Precedence

Settings are applied in this order (later overrides earlier):

1. **Default values** (hardcoded in the code)
2. **Environment variables** (from `.env` file)
3. **Command-line arguments** (passed directly to scripts)
4. **Config files** (YAML files in `config/` directory)

Example:
```bash
# Default is gpt-4, but .env overrides to gpt-3.5-turbo
# Command line overrides both:
python examples/quantopia_demo.py --model gpt-4-turbo
```

## Loading Environment Variables

### Method 1: Automatic (Recommended)

Most scripts automatically load `.env` using python-dotenv:

```python
from dotenv import load_dotenv
load_dotenv()  # Automatically loads .env file
```

### Method 2: Manual Export

```bash
# Export for current shell session
export OPENAI_API_KEY="sk-your-key"

# Or load from .env file
export $(cat .env | xargs)
```

### Method 3: Shell Integration

Add to `~/.zshrc` or `~/.bashrc`:

```bash
# Auto-load .env when entering project directory
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi
```

## Security Best Practices

### DO ✅
- ✅ Use `.env` file for local development
- ✅ Add `.env` to `.gitignore`
- ✅ Use separate keys for dev/staging/prod
- ✅ Rotate keys periodically
- ✅ Set spending limits on API provider dashboards
- ✅ Use read-only keys when possible

### DON'T ❌
- ❌ Commit `.env` or API keys to git
- ❌ Share keys in Slack/Discord/email
- ❌ Use production keys for development
- ❌ Hardcode keys in source code
- ❌ Use the same key across multiple projects

## Checking Your Configuration

```bash
# Quick check: verify API key is loaded
python -c "
import os
from dotenv import load_dotenv
load_dotenv()

key = os.getenv('OPENAI_API_KEY')
if key:
    print(f'✅ API key loaded: {key[:8]}...{key[-4:]}')
else:
    print('❌ No API key found!')
"

# Test API connection
python examples/quick_test.py
```

## Troubleshooting

### "API key not found"

```bash
# Check if .env exists
ls -la .env

# Check if key is in .env
grep OPENAI_API_KEY .env

# Verify it's loaded
echo $OPENAI_API_KEY

# If empty, reload:
source venv/bin/activate
export $(cat .env | xargs)
```

### "Permission denied"

```bash
# Make sure .env has correct permissions
chmod 600 .env
```

### "API key invalid"

1. Verify key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Check for extra spaces or quotes
3. Ensure key starts with `sk-`
4. Try regenerating the key

## Environment-Specific Configurations

### Development
```bash
# .env.development
OPENAI_API_KEY=sk-dev-key
LOG_LEVEL=DEBUG
NUM_STRATEGIES_PER_BATCH=2
```

### Production
```bash
# .env.production
OPENAI_API_KEY=sk-prod-key
LOG_LEVEL=INFO
NUM_STRATEGIES_PER_BATCH=10
DATABASE_PATH=/var/lib/quantopia/prod.db
```

Load specific environment:
```bash
cp .env.production .env
# or
export ENV=production
```

## Next Steps

After setting up your environment:

1. ✅ Run `python examples/quick_test.py` to verify API works
2. ✅ See [GETTING_STARTED_REAL_API.md](GETTING_STARTED_REAL_API.md) for full workflow
3. ✅ Review [.env.example](.env.example) for all options
4. ✅ Run your first iteration!

Happy trading! 🚀
