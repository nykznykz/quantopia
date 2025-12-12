"""LLM-based code generator for converting strategy metadata to executable Python classes."""

from typing import Dict, Any, List, Optional
import json
import ast
import logging

from src.strategy_generation.llm_client import LLMClient

logger = logging.getLogger(__name__)


class CodeValidationError(Exception):
    """Raised when generated code fails validation."""
    pass


class CodeGenerator:
    """Generates executable Python code from strategy metadata using LLM."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_retries: int = 3
    ):
        """Initialize LLM-based code generator.

        Args:
            llm_client: LLM client instance (creates default if None)
            model: Model name override (e.g., "gpt-4", "claude-sonnet-4.5")
            temperature: LLM temperature (0.2 = more deterministic for code)
            max_retries: Maximum retries for code generation
        """
        if llm_client is None:
            # Create default client with coding-optimized settings
            self.llm_client = LLMClient(
                provider="openai",
                model=model or "gpt-4",
                temperature=temperature
            )
        else:
            self.llm_client = llm_client
            # Override temperature for code generation
            self.llm_client.temperature = temperature

        self.max_retries = max_retries

        logger.info(
            f"Initialized LLM-based CodeGenerator "
            f"(model={self.llm_client.model}, temp={temperature})"
        )

    def generate_strategy_class(self, strategy_metadata: Dict[str, Any]) -> str:
        """Generate complete Python class from strategy metadata using LLM.

        Args:
            strategy_metadata: Strategy dict from LLM strategy generator

        Returns:
            Python code as string

        Raises:
            CodeValidationError: If code generation fails after retries
        """
        strategy_name = strategy_metadata.get('strategy_name', 'Unknown')

        logger.info(f"Generating code for strategy: {strategy_name}")

        # Build prompt for code generation
        prompt = self._create_code_generation_prompt(strategy_metadata)

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Code generation attempt {attempt}/{self.max_retries}")

                # Generate code with LLM
                code = self.llm_client.generate(
                    prompt=prompt,
                    system_prompt=self._get_system_prompt(),
                    temperature=self.llm_client.temperature
                )

                # Clean up code (remove markdown fences if present)
                code = self._clean_code_output(code)

                # Validate generated code
                self._validate_code(code, strategy_metadata)

                logger.info(f"✓ Successfully generated code for: {strategy_name}")
                logger.info(f"  Code length: {len(code)} characters, {len(code.splitlines())} lines")

                return code

            except (CodeValidationError, SyntaxError, Exception) as e:
                last_error = e
                logger.warning(f"Attempt {attempt} failed: {e}")

                if attempt < self.max_retries:
                    # Add feedback for retry
                    prompt += f"\n\nPrevious attempt failed with error:\n{e}\n\nPlease fix the code and try again. Ensure the code is syntactically correct and follows all requirements."

        # All retries failed
        raise CodeValidationError(
            f"Failed to generate valid code for {strategy_name} after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def _get_system_prompt(self) -> str:
        """Get system prompt for code generation."""
        return """You are an expert Python developer specializing in algorithmic trading systems.

Your task is to generate clean, efficient, production-ready Python code for trading strategies.

Requirements:
1. Generate ONLY the Python code, no markdown fences or explanations
2. Code must be syntactically correct and executable
3. Follow PEP 8 style guidelines
4. Include proper error handling
5. Use type hints where appropriate
6. Add clear, concise docstrings
7. The generated class MUST inherit from BaseStrategy
8. All indicators must be accessed from INDICATOR_REGISTRY
9. Use self.indicators dict to store calculated indicator values
10. Implement all required methods: __init__, calculate_indicators, should_enter, should_exit

CRITICAL PERFORMANCE REQUIREMENT:
- calculate_indicators() must return FULL SERIES (pd.Series/pd.DataFrame), NOT single values
- Do NOT use .iloc[-1] to extract last values in calculate_indicators()
- The backtest engine pre-calculates indicators once and indexes into them
- In should_enter()/should_exit(), self.indicators will contain scalar values (already extracted)

Generate clean, production-ready code that can be executed immediately."""

    def _create_code_generation_prompt(self, strategy_metadata: Dict[str, Any]) -> str:
        """Create prompt for LLM code generation.

        Args:
            strategy_metadata: Strategy metadata dict

        Returns:
            Formatted prompt string
        """
        # Import indicator specifications
        from src.indicators.specs import INDICATOR_SPECS_STRING

        # Format strategy metadata as JSON
        strategy_json = json.dumps(strategy_metadata, indent=2)

        # Check if ML model is involved
        has_ml = strategy_metadata.get('ml_strategy_type') != 'pure_technical'
        ml_model_id = strategy_metadata.get('ml_model_id')

        prompt = f"""Generate a complete Python trading strategy class based on this specification:

{strategy_json}

{INDICATOR_SPECS_STRING}"""

        # Add ML-specific instructions if needed
        if has_ml and ml_model_id:
            prompt += f"""

ML MODEL INTEGRATION:
This strategy uses ML model: {ml_model_id}

The strategy class must include ML model loading and prediction:
1. Load the model in __init__() using self.load_ml_model()
2. Get ML predictions in calculate_indicators() using self.get_ml_signal()
3. Use ML predictions as signals in should_enter() and should_exit()

Example ML integration:
```python
def __init__(self, exchange, price_feed, symbol, initial_capital=10000.0):
    super().__init__(exchange, price_feed, symbol, initial_capital)

    # Load ML model
    self.load_ml_model(
        model_path="models/{ml_model_id}.joblib",
        pipeline_path="models/{ml_model_id}_pipeline.joblib"
    )

    # Initialize regular indicators...

def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
    indicators = {{}}

    # Calculate regular indicators first...

    # Get ML prediction
    ml_signal = self.get_ml_signal(data, threshold=0.6)
    indicators['ml_signal'] = ml_signal  # 1 (bullish), -1 (bearish), 0 (neutral)

    return indicators

def should_enter(self) -> bool:
    # Combine ML signal with technical indicators
    ml_bullish = self.indicators.get('ml_signal', 0) == 1
    rsi_condition = self.indicators.get('rsi', 50) < 40

    return ml_bullish and rsi_condition  # Both conditions
```

IMPORTANT: The ML signal should be treated as one component of the entry/exit decision, not the only factor."""

        prompt += """

The class must follow this structure:

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


class YourStrategyName(BaseStrategy):
    \"\"\"
    [Strategy Name]

    Type: [strategy_type]

    Hypothesis:
    [hypothesis from metadata]
    \"\"\"

    def __init__(self, exchange, price_feed, symbol, initial_capital=10000.0):
        \"\"\"Initialize strategy.\"\"\"
        super().__init__(exchange, price_feed, symbol, initial_capital)

        # Initialize indicator instances from INDICATOR_REGISTRY
        # Example: self.rsi = INDICATOR_REGISTRY['RSI'](period=14)
        # [Initialize all indicators from metadata]

    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        \"\"\"Calculate all strategy indicators.

        IMPORTANT: This method should return FULL SERIES for all indicators, not just the last value.
        The backtest engine will index into these series for each bar.

        Args:
            data: Historical OHLCV DataFrame

        Returns:
            Dict with indicator values as pd.Series or pd.DataFrame
        \"\"\"
        indicators = {{}}

        # Calculate each indicator and return the FULL SERIES
        # Example:
        #   rsi_series = self.rsi.calculate(data)  # Returns full series
        #   indicators['rsi'] = rsi_series  # Store the full series, NOT rsi_series.iloc[-1]
        #
        # For multi-value indicators (MACD, Bollinger Bands, ADX), extract columns but keep as series:
        #   macd_df = self.macd.calculate(data)  # Returns DataFrame
        #   indicators['macd'] = macd_df['macd']  # Store series, NOT .iloc[-1]
        #   indicators['signal'] = macd_df['signal']
        #   indicators['histogram'] = macd_df['histogram']

        # Store current price series
        if 'close' in data.columns and len(data) > 0:
            indicators['current_price'] = data['close']  # Full series, NOT .iloc[-1]

        return indicators

    def should_enter(self) -> bool:
        \"\"\"Determine if should enter position.

        Returns:
            True if entry conditions met
        \"\"\"
        # Check if we have all required indicators
        required_indicators = [k for k in self.indicators.keys() if k != "current_price"]
        if not all(self.indicators.get(ind) is not None for ind in required_indicators):
            return False

        try:
            # Implement entry rules from metadata
            # Example: return self.indicators['rsi'] < 30 and self.indicators['current_price'] < self.indicators['ema']
            pass
        except Exception as e:
            return False

    def should_exit(self) -> bool:
        \"\"\"Determine if should exit position.

        Returns:
            True if exit conditions met
        \"\"\"
        if not self.indicators:
            return False

        try:
            # Implement exit rules from metadata
            pass
        except Exception as e:
            return False

    def stop_loss(self, entry_price: float) -> Optional[float]:
        \"\"\"Calculate stop loss price.

        Args:
            entry_price: Entry price of position

        Returns:
            Stop loss price or None
        \"\"\"
        # Implement stop loss from metadata
        pass

    def take_profit(self, entry_price: float) -> Optional[float]:
        \"\"\"Calculate take profit price.

        Args:
            entry_price: Entry price of position

        Returns:
            Take profit price or None
        \"\"\"
        # Implement take profit from metadata
        pass
```

Important notes:
- **CRITICAL**: Use EXACT parameter names from the indicator specifications above - never infer or guess parameter names
- **CRITICAL**: Look up each indicator in the specs to see its parameters, return type, and usage example
- Indicator names in the metadata should be accessed from INDICATOR_REGISTRY
- For indicators with parameters, pass them during initialization using the EXACT parameter names from specs
- Multi-output indicators (MACD, BollingerBands, Stochastic, ADX, etc.) return DataFrames - extract specific columns
- Single-output indicators (RSI, EMA, SMA, ATR, etc.) return Series - use iloc[-1] to get latest value
- Include proper error handling in all methods

**CRITICAL IMPLEMENTATION INSTRUCTIONS - YOUR ONLY JOB IS TRANSLATION, NOT STRATEGY DESIGN:**

1. **Faithfully implement the EXACT logic provided in entry_conditions.logic and exit_conditions.logic**:
   - The strategy specification contains complete boolean expressions (e.g., "(RSI(14) < 44 OR price < BB_lower) AND volume > avg_volume")
   - Translate these EXACTLY to Python - do NOT modify thresholds, do NOT add conditions, do NOT change AND/OR logic
   - Use the EXACT numeric thresholds specified (e.g., if logic says "< 44", use < 44, not < 30 or < 40)

2. **Do NOT make strategic decisions**:
   - Do NOT decide how to combine conditions (the logic string specifies AND/OR explicitly)
   - Do NOT modify or "improve" thresholds (they are calibrated to market statistics)
   - Do NOT add additional conditions beyond what the logic string specifies
   - Do NOT interpret vague terms - the logic is specific and complete

3. **Translation approach**:
   - Parse the logic string to identify all conditions
   - Map indicator names to self.indicators dict keys (lowercase, no periods)
   - Implement the boolean expression with proper Python syntax
   - Handle None/NaN values gracefully (return False if key indicators are None)

4. **Example translation**:
   ```python
   # Logic string: "(RSI(14) < 44 OR RSI(5) < 35) AND TrendStrength < 0.5"

   def should_enter(self) -> bool:
       rsi_14 = self.indicators.get('rsi_14')
       rsi_5 = self.indicators.get('rsi_5')
       trend_strength = self.indicators.get('trendstrength')

       # Check for None values
       if None in [rsi_14, rsi_5, trend_strength]:
           return False

       # Direct translation of logic string
       oversold = (rsi_14 < 44) or (rsi_5 < 35)
       ranging = trend_strength < 0.5

       return oversold and ranging
   ```

5. **For risk management**:
   - Implement stop_loss and take_profit based on risk_management specifications
   - Use exact values/formulas specified (e.g., "2.0 * ATR(14)" means multiply ATR by 2.0)

Example of CORRECT indicator initialization:
```python
# CORRECT - uses exact parameter name from specs
self.roc = INDICATOR_REGISTRY['ROC'](period=10)  # ✓ 'period' is correct

# WRONG - inferred parameter name
self.roc = INDICATOR_REGISTRY['ROC'](lookback=10)  # ✗ 'lookback' doesn't exist
```

Generate the complete, executable Python code now."""

        return prompt

    def _clean_code_output(self, code: str) -> str:
        """Clean up LLM code output (remove markdown fences, etc.).

        Args:
            code: Raw code output from LLM

        Returns:
            Cleaned code string
        """
        # Remove markdown code fences
        if code.startswith("```python"):
            code = code[len("```python"):].lstrip()
        elif code.startswith("```"):
            code = code[len("```"):].lstrip()

        if code.endswith("```"):
            code = code[:-len("```")].rstrip()

        # Remove any leading/trailing whitespace
        code = code.strip()

        return code

    def _validate_code(self, code: str, strategy_metadata: Dict[str, Any]) -> None:
        """Validate generated code.

        Args:
            code: Generated Python code
            strategy_metadata: Original strategy metadata

        Raises:
            CodeValidationError: If validation fails
        """
        # 1. Check if code is not empty
        if not code or len(code.strip()) == 0:
            raise CodeValidationError("Generated code is empty")

        # 2. Syntax check
        try:
            ast.parse(code)
        except SyntaxError as e:
            raise CodeValidationError(f"Syntax error in generated code: {e}")

        # 3. Check for required imports
        required_imports = [
            "BaseStrategy",
            "INDICATOR_REGISTRY",
            "pd.DataFrame"
        ]
        for import_name in required_imports:
            if import_name not in code:
                raise CodeValidationError(f"Missing required import or reference: {import_name}")

        # 4. Check for required methods
        required_methods = [
            "def __init__",
            "def calculate_indicators",
            "def should_enter",
            "def should_exit",
            "def stop_loss",
            "def take_profit"
        ]
        for method in required_methods:
            if method not in code:
                raise CodeValidationError(f"Missing required method: {method}")

        # 5. Check that class inherits from BaseStrategy
        if "BaseStrategy" not in code or "(BaseStrategy)" not in code:
            raise CodeValidationError("Class must inherit from BaseStrategy")

        # 6. Check that indicators from metadata are referenced
        indicator_names = [ind.get('name') for ind in strategy_metadata.get('indicators', [])]
        missing_indicators = []
        for indicator_name in indicator_names:
            # Check if indicator is initialized in __init__
            if f"INDICATOR_REGISTRY['{indicator_name}']" not in code and \
               f'INDICATOR_REGISTRY["{indicator_name}"]' not in code:
                missing_indicators.append(indicator_name)

        if missing_indicators:
            logger.warning(
                f"Some indicators from metadata not found in code: {missing_indicators}"
            )
            # Don't fail validation for this, just warn

        logger.info("✓ Code validation passed")


def save_strategy_code(code: str, filepath: str):
    """Save generated strategy code to file.

    Args:
        code: Python code string
        filepath: Output file path
    """
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w') as f:
        f.write(code)

    logger.info(f"Saved strategy code to: {filepath}")
