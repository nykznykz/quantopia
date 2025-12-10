# Getting Started with Real API Calls

This guide will help you set up and run Quantopia with real LLM API calls.

## Step 1: Get API Keys

### Option A: OpenAI (Recommended for Getting Started)

1. **Create an account** at https://platform.openai.com/
2. **Add payment method** (required for API access)
3. **Generate API key**:
   - Go to https://platform.openai.com/api-keys
   - Click "Create new secret key"
   - Copy and save the key securely (you won't see it again!)

**Cost estimate**:
- GPT-4: ~$0.03/1K input tokens, ~$0.06/1K output tokens
- For 10 strategies: Roughly $2-5 per iteration (depends on complexity)

### Option B: Anthropic Claude (Best for Code Generation)

1. **Create account** at https://console.anthropic.com/
2. **Add payment method**
3. **Generate API key**:
   - Go to https://console.anthropic.com/settings/keys
   - Create new key

**Cost estimate**:
- Claude Sonnet 4.5: ~$3/MTok input, ~$15/MTok output
- Generally cheaper than GPT-4 for code generation

### Option C: DeepSeek (Budget Option)

1. **Create account** at https://platform.deepseek.com/
2. **Generate API key**

**Cost estimate**:
- DeepSeek-Coder: ~$0.14/MTok
- Much cheaper, good for testing

## Step 2: Set Up Environment Variables

### On macOS/Linux:

```bash
# Add to ~/.bashrc or ~/.zshrc
export OPENAI_API_KEY="sk-your-key-here"
export ANTHROPIC_API_KEY="sk-ant-your-key-here"  # optional

# Reload shell
source ~/.bashrc  # or source ~/.zshrc
```

### Alternative: Use .env File

```bash
# Create .env file in project root
cd /Users/user/Documents/quant/quantopia
cat > .env << 'ENVFILE'
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
ENVFILE

# Add to .gitignore to prevent committing secrets
echo ".env" >> .gitignore
```

## Step 3: Verify API Access

```bash
# Activate venv
source venv/bin/activate

# Test OpenAI connection
python -c "
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model='gpt-4',
    messages=[{'role': 'user', 'content': 'Hello!'}],
    max_tokens=10
)
print('OpenAI API working!')
print(response.choices[0].message.content)
"
```

## Step 4: Create a Simple Test Script

```bash
# Create examples/quick_test.py
cat > examples/quick_test.py << 'PYFILE'
#!/usr/bin/env python3
"""Quick test of strategy generation with real API."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategy_generation.llm_client import LLMClient
from src.strategy_generation.generator import StrategyGenerator
from src.indicators.indicator_library import AVAILABLE_INDICATORS

def main():
    print("Initializing LLM client...")
    
    # Initialize LLM client
    llm_client = LLMClient(
        provider="openai",
        model="gpt-4",
        temperature=0.7
    )
    
    # Initialize strategy generator
    available_indicators = list(AVAILABLE_INDICATORS.keys())
    generator = StrategyGenerator(
        llm_client=llm_client,
        available_indicators=available_indicators
    )
    
    # Generate a single strategy
    print("\nGenerating strategy...")
    try:
        strategy = generator.generate_strategy(
            strategy_type="momentum",
            asset_class="crypto",
            timeframe="1h"
        )
        
        print(f"\n{'='*80}")
        print(f"Generated Strategy: {strategy['strategy_name']}")
        print(f"{'='*80}")
        print(f"Type: {strategy['strategy_type']}")
        print(f"Indicators: {', '.join(strategy['indicators'])}")
        print(f"\nEntry Conditions:")
        print(f"  {strategy['entry_conditions']}")
        print(f"\nExit Conditions:")
        print(f"  {strategy['exit_conditions']}")
        print(f"\nPosition Sizing: {strategy['position_sizing']}")
        print(f"Stop Loss: {strategy.get('stop_loss_pct', 'N/A')}")
        print(f"Take Profit: {strategy.get('take_profit_pct', 'N/A')}")
        print(f"{'='*80}")
        
        print("\n✅ Success! API is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check that OPENAI_API_KEY is set: echo $OPENAI_API_KEY")
        print("2. Verify your API key is valid at https://platform.openai.com/api-keys")
        print("3. Ensure you have credits/payment method on your OpenAI account")
        sys.exit(1)

if __name__ == '__main__':
    main()
PYFILE

chmod +x examples/quick_test.py
```

## Step 5: Run Your First Test

```bash
# Activate venv
source venv/bin/activate

# Set your API key
export OPENAI_API_KEY="sk-your-actual-key-here"

# Run quick test
python examples/quick_test.py
```

**Expected output:**
```
Initializing LLM client...

Generating strategy...

================================================================================
Generated Strategy: RSI_Momentum_1h_v1
================================================================================
Type: momentum
Indicators: rsi, ema_20, volume

Entry Conditions:
  RSI crosses above 30 AND price above EMA(20) AND volume > 1.5x average

Exit Conditions:
  RSI crosses above 70 OR price crosses below EMA(20)

Position Sizing: 0.1
Stop Loss: 0.02
Take Profit: 0.04
================================================================================

✅ Success! API is working correctly.
```

## Step 6: Run a Full Iteration (Start Small!)

```bash
# Create sample data first
mkdir -p data

python -c "
from examples.quantopia_demo import create_sample_data
data = create_sample_data(days=30, freq='1h')
data.to_csv('data/btc_sample_30d.csv', index=False)
print('Sample data created at data/btc_sample_30d.csv')
"

# Run demo with just 2 strategies to start
python examples/quantopia_demo.py \
    --api-key $OPENAI_API_KEY \
    --iterations 1 \
    --strategies-per-batch 2 \
    --symbol BTC-USD \
    --data-days 30 \
    --database data/quantopia_test.db \
    --export results/first_run.json

# This will take 2-5 minutes and cost ~$0.20-0.50
```

## Step 7: Scale Up Gradually

```bash
# Once comfortable, increase batch size
python examples/quantopia_demo.py \
    --api-key $OPENAI_API_KEY \
    --iterations 1 \
    --strategies-per-batch 5 \
    --symbol BTC-USD \
    --database data/quantopia_test.db

# Then try multiple iterations
python examples/quantopia_demo.py \
    --api-key $OPENAI_API_KEY \
    --iterations 3 \
    --strategies-per-batch 5 \
    --symbol BTC-USD \
    --database data/quantopia_prod.db
```

## Cost Estimation

### Small Test (2 strategies, 1 iteration)
- **Cost**: ~$0.20-0.40
- **Time**: 2-3 minutes
- **Good for**: Initial testing

### Medium Run (5 strategies, 1 iteration)
- **Cost**: ~$1-2
- **Time**: 5-8 minutes
- **Good for**: Regular development

### Production Run (10 strategies, 3 iterations)
- **Cost**: ~$6-12
- **Time**: 20-30 minutes
- **Good for**: Actual strategy discovery

## Troubleshooting

### "API key not found"
```bash
# Verify it's set
echo $OPENAI_API_KEY

# If empty, set it
export OPENAI_API_KEY="sk-your-key-here"

# Or add to ~/.zshrc for persistence
echo 'export OPENAI_API_KEY="sk-your-key-here"' >> ~/.zshrc
source ~/.zshrc
```

### "Rate limit exceeded"
- Wait 60 seconds and try again
- Reduce batch size: `--strategies-per-batch 2`
- The system has built-in retry logic

### "Insufficient credits"
- Add payment method at https://platform.openai.com/account/billing
- Check usage at https://platform.openai.com/usage

## Best Practices

1. **Start small**: 2-3 strategies first
2. **Monitor costs**: Check OpenAI dashboard
3. **Save results**: Always use `--export` flag
4. **Use sample data**: Test with synthetic data first
5. **Check logs**: Review output for errors

## What's Next?

Once you've run your first successful iteration:

1. **Analyze results** in the database:
   ```bash
   sqlite3 data/quantopia_test.db "SELECT * FROM strategies LIMIT 5;"
   ```

2. **Review generated code**:
   ```bash
   sqlite3 data/quantopia_test.db "SELECT code FROM strategy_code LIMIT 1;" | head -50
   ```

3. **Check backtest results**:
   ```bash
   sqlite3 data/quantopia_test.db "SELECT strategy_id, sharpe_ratio, total_return FROM backtest_results ORDER BY sharpe_ratio DESC LIMIT 10;"
   ```

4. **Scale up** to production runs

Happy trading! 🚀
