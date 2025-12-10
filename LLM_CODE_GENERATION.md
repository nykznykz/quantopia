# LLM-Based Code Generation

## Overview

OmniAlpha uses **LLM-based code generation** instead of template-based heuristics. The system employs a **two-stage LLM process**:

1. **Stage 1**: LLM generates strategy metadata (JSON) - strategy logic, indicators, rules
2. **Stage 2**: LLM generates executable Python code from the metadata

This approach provides maximum flexibility while maintaining code quality through validation.

## Architecture

```
┌──────────────────────────────────────────────┐
│  Stage 1: Strategy Design (LLM)             │
│                                              │
│  Input: "Generate mean reversion strategy"  │
│  Output: JSON metadata                       │
│    - indicators: [RSI, EMA]                  │
│    - entry_rules: ["RSI < 30", ...]         │
│    - exit_rules: ["RSI > 70", ...]          │
└──────────────┬───────────────────────────────┘
               │ Structured JSON
               ▼
┌──────────────────────────────────────────────┐
│  Stage 2: Code Generation (LLM)             │
│                                              │
│  Input: Strategy JSON + Code prompt         │
│  Output: Executable Python class            │
│    - Imports and setup                       │
│    - Indicator initialization                │
│    - Entry/exit logic                        │
│    - Risk management                         │
└──────────────┬───────────────────────────────┘
               │ Python code
               ▼
┌──────────────────────────────────────────────┐
│  Code Validation                             │
│  - Syntax check (AST parsing)               │
│  - Required methods check                    │
│  - Indicator validation                      │
│  - Logic completeness check                  │
└──────────────┬───────────────────────────────┘
               │ Validated code
               ▼
┌──────────────────────────────────────────────┐
│  Backtest Execution (SimulatedExchange)     │
└──────────────────────────────────────────────┘
```

## Why LLM-Based Code Generation?

### Advantages over Template-Based

1. **Flexibility**: Can generate complex logic that templates can't express
2. **Naturalness**: Generates idiomatic Python code
3. **Adaptability**: Can handle edge cases and unusual strategies
4. **Extensibility**: No need to update templates for new patterns
5. **Documentation**: Generates helpful comments and docstrings

### Validation Ensures Quality

- Syntax checking (AST parsing)
- Required method validation
- Indicator reference checking
- Logic completeness verification
- Optional: Import testing

## Recommended Models for Code Generation

### OpenAI (GPT-4)
```python
llm_client = LLMClient(
    provider="openai",
    model="gpt-4",
    temperature=0.2,  # Low for deterministic code
    max_tokens=4000
)
```

**Pros**: Widely available, good code quality, fast
**Cost**: ~$0.01-0.05 per strategy

### Anthropic (Claude Sonnet 4)
```python
llm_client = LLMClient(
    provider="anthropic",
    model="claude-sonnet-4",
    temperature=0.2,
    max_tokens=4096
)
```

**Pros**: Excellent reasoning, very good at code, strong with complex logic
**Cost**: ~$0.015-0.06 per strategy
**Best for**: Complex strategies requiring sophisticated logic

### DeepSeek (DeepSeek Coder)
```python
llm_client = LLMClient(
    provider="deepseek",
    model="deepseek-coder",
    temperature=0.2,
    max_tokens=4000
)
```

**Pros**: Code-specialized, very cost-effective (~10x cheaper)
**Cost**: ~$0.001-0.005 per strategy
**Best for**: Batch processing, experimentation

## Usage

### Basic Usage

```python
from src.indicators import INDICATOR_REGISTRY
from src.strategy_generation import StrategyGenerator, LLMClient
from src.code_generation import CodeGenerator

# 1. Generate strategy metadata
llm_client = LLMClient(provider="openai", model="gpt-4", temperature=0.8)
generator = StrategyGenerator(llm_client, list(INDICATOR_REGISTRY.keys()))
strategy_metadata = generator.generate_strategy(strategy_type="mean_reversion")

# 2. Generate code with LLM (using same or different client)
code_gen = CodeGenerator(
    llm_client=llm_client,  # Can reuse same client
    temperature=0.2  # Lower temp for code generation
)
strategy_code = code_gen.generate_strategy_class(strategy_metadata)

# Code is now executable Python!
```

### Using Different Models for Different Stages

You can use different models for strategy generation vs code generation:

```python
# Strategy generation: Use creative model
strategy_llm = LLMClient(provider="openai", model="gpt-4", temperature=0.9)
generator = StrategyGenerator(strategy_llm, indicators)
strategy = generator.generate_strategy()

# Code generation: Use code-specialized model
code_llm = LLMClient(provider="anthropic", model="claude-sonnet-4", temperature=0.2)
code_gen = CodeGenerator(llm_client=code_llm)
code = code_gen.generate_strategy_class(strategy)
```

### Helper: Create Optimized Client

```python
from src.strategy_generation.llm_client import create_code_generation_client

# Automatically configured for code generation
code_llm = create_code_generation_client(provider="anthropic")
# Uses recommended model, temperature=0.2, max_tokens=4096
```

## Code Generation Process

### 1. Prompt Construction

The CodeGenerator builds a detailed prompt including:
- Strategy metadata (JSON)
- Required class structure
- Import statements
- Method signatures
- Examples and guidelines

### 2. LLM Generation

The LLM generates complete Python code following the prompt structure.

### 3. Cleaning

- Remove markdown code fences (`\`\`\`python`)
- Strip extra whitespace
- Clean formatting

### 4. Validation

Multiple validation layers:
- **Syntax**: AST parsing to ensure valid Python
- **Structure**: Check for BaseStrategy inheritance
- **Methods**: Verify all required methods exist
- **Indicators**: Confirm indicators from metadata are initialized
- **Logic**: Ensure entry/exit methods have actual logic (not just `pass`)

### 5. Retry Logic

If validation fails, the system:
1. Adds error feedback to prompt
2. Retries with corrected prompt
3. Maximum 3 attempts by default

## Example Generated Code

From this metadata:
```json
{
  "strategy_name": "RSI Mean Reversion",
  "indicators": [
    {"name": "RSI", "params": {"period": 14}},
    {"name": "EMA", "params": {"period": 20}}
  ],
  "entry_rules": ["RSI < 30", "price < EMA"],
  "exit_rules": ["RSI > 70"]
}
```

The LLM generates:
```python
import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../simulated_exchange/src"))

from src.code_generation.strategy_base import BaseStrategy
from src.indicators import INDICATOR_REGISTRY


class RsiMeanReversionStrategy(BaseStrategy):
    """
    RSI Mean Reversion Strategy

    Type: mean_reversion

    Hypothesis:
    When RSI falls below 30 and price is below EMA, market is oversold
    and likely to revert upward.
    """

    def __init__(self, exchange, price_feed, symbol, initial_capital=10000.0):
        """Initialize strategy."""
        super().__init__(exchange, price_feed, symbol, initial_capital)

        # Initialize indicators
        self.rsi = INDICATOR_REGISTRY['RSI'](period=14)
        self.ema = INDICATOR_REGISTRY['EMA'](period=20)

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all strategy indicators."""
        indicators = {}

        # Calculate RSI
        rsi_values = self.rsi.calculate(data)
        if len(rsi_values) > 0:
            indicators['rsi'] = rsi_values.iloc[-1]
        else:
            indicators['rsi'] = None

        # Calculate EMA
        ema_values = self.ema.calculate(data)
        if len(ema_values) > 0:
            indicators['ema'] = ema_values.iloc[-1]
        else:
            indicators['ema'] = None

        # Store current price
        if 'close' in data.columns and len(data) > 0:
            indicators['current_price'] = data['close'].iloc[-1]

        return indicators

    def should_enter(self) -> bool:
        """Determine if should enter position."""
        # Check we have required indicators
        required_indicators = [k for k in self.indicators.keys() if k != "current_price"]
        if not all(self.indicators.get(ind) is not None for ind in required_indicators):
            return False

        try:
            # Entry conditions
            rsi_condition = self.indicators['rsi'] < 30
            price_condition = self.indicators['current_price'] < self.indicators['ema']

            return rsi_condition and price_condition
        except Exception as e:
            return False

    def should_exit(self) -> bool:
        """Determine if should exit position."""
        if not self.indicators:
            return False

        try:
            # Exit condition
            return self.indicators['rsi'] > 70
        except Exception as e:
            return False

    def stop_loss(self, entry_price: float) -> Optional[float]:
        """Calculate stop loss price."""
        return entry_price * 0.98  # 2% stop loss

    def take_profit(self, entry_price: float) -> Optional[float]:
        """Calculate take profit price."""
        return entry_price * 1.05  # 5% take profit
```

## Configuration

### Temperature Settings

**For Strategy Generation** (creativity):
- `0.7-0.9`: More creative, diverse strategies
- `0.5-0.7`: Balanced
- `0.3-0.5`: Conservative, safer strategies

**For Code Generation** (determinism):
- `0.1-0.3`: Highly consistent code (recommended)
- `0.3-0.5`: Some variation in style
- `0.5+`: Not recommended (too variable)

### Max Tokens

- **Minimum**: 2000 tokens
- **Recommended**: 4000 tokens
- **Complex strategies**: 6000+ tokens

### Retries

Default: 3 attempts
- Increase for complex strategies
- Decrease to save cost/time

## Error Handling

### Common Errors and Solutions

1. **"Missing required method"**
   - LLM didn't generate all methods
   - Solution: Retry with more explicit prompt

2. **"Syntax error"**
   - LLM generated invalid Python
   - Solution: Automatic retry with error feedback

3. **"No logic in should_enter"**
   - LLM only generated `pass` statement
   - Solution: Retry with emphasis on implementing logic

4. **"Indicator not found in INDICATOR_REGISTRY"**
   - LLM hallucinated an indicator name
   - Solution: Retry with list of valid indicators

## Performance

### Speed
- **Template-based**: < 1 second
- **LLM-based**: 10-30 seconds

### Cost (per strategy)
- **OpenAI GPT-4**: $0.01-0.05
- **Claude Sonnet 4**: $0.015-0.06
- **DeepSeek Coder**: $0.001-0.005

### Quality
- **Template-based**: 100% syntactically correct, limited logic
- **LLM-based**: 95%+ success rate, flexible logic, may require retries

## Best Practices

1. **Use low temperature for code** (0.2-0.3)
2. **Provide clear strategy metadata** (detailed rules)
3. **Enable retries** (at least 3 attempts)
4. **Validate before backtesting** (automatic)
5. **Review generated code** (spot-check periodically)
6. **Choose model based on needs**:
   - GPT-4: General purpose, widely available
   - Claude: Complex logic, best reasoning
   - DeepSeek: Batch processing, cost-sensitive

## Extending the System

### Add Custom Validation

```python
from src.code_generation import CodeValidator

validator = CodeValidator()
is_valid, errors = validator.validate_comprehensive(code, strategy_metadata)
```

### Custom Code Generation Prompts

Modify `CodeGenerator._create_code_generation_prompt()` to customize the prompt structure.

### Add New LLM Providers

Add support in `LLMClient.__init__()` following the existing pattern.

## Troubleshooting

### Code generation fails repeatedly
- Check API key is valid
- Verify model name is correct
- Try increasing max_tokens
- Check if strategy metadata is valid

### Generated code doesn't compile
- Validation should catch this
- If it slips through, report as bug
- Increase validation rigor

### Performance is too slow
- Use faster model (DeepSeek)
- Reduce max_tokens
- Batch multiple strategies

## Future Enhancements

Potential improvements:
- Fine-tuned models on trading code
- Few-shot examples in prompts
- Code optimization passes
- Automated refactoring
- Performance profiling of generated code

## Conclusion

LLM-based code generation provides **maximum flexibility** while maintaining **quality through validation**. The two-stage approach (strategy design → code generation) separates concerns and allows using different models for different tasks.

**Trade-off**: Slightly slower and more expensive than templates, but much more capable and extensible.
