# Architectural Fix: Proper Separation of Concerns

## Problem Identified

**Current (Flawed) Design:**
```
Strategy Agent → generates initial strategies
       ↓
   Backtest → fails (0 trades)
       ↓
Critique Agent → analyzes failure AND generates refined strategy ❌ WRONG
```

**Issue:** Critique Agent is generating strategies, violating separation of concerns.

---

## Root Cause Analysis

### Does Strategy Agent Currently Learn from Failures?

Looking at `src/agents/strategy_agent.py:91-127`, the `_gather_context()` method queries:

✅ **What it DOES query:**
- Top performers (successes)
- Recent strategies (avoid duplicates)
- Underexplored areas

❌ **What it DOESN'T query:**
- Failed strategies (0 trades, low Sharpe, rejected)
- Failure patterns across multiple strategies
- Specific reasons for failure (too restrictive, too loose, etc.)

**Verdict:** Strategy Agent looks at successes and gaps, but **does NOT systematically learn from failures**.

---

## Proper Architecture (Fixed)

### Separation of Concerns

| Component | Responsibility | Does NOT Do |
|-----------|----------------|-------------|
| **Strategy Agent** | - Queries database for ALL history (successes + failures)<br>- Analyzes failure patterns<br>- Generates ALL strategies (new + refinements)<br>- Decides what to try next | - Does NOT evaluate backtest results<br>- Does NOT filter strategies |
| **Critique Agent** | - Analyzes backtest results<br>- Provides structured feedback<br>- Identifies failure root causes<br>- Suggests improvement directions | - Does NOT generate strategy specs<br>- Does NOT decide what to try next<br>- Only provides analysis |
| **Filter** | - Evaluates metrics (Sharpe, DD, trades)<br>- Approves/rejects strategies | - Does NOT analyze WHY it failed<br>- Just applies thresholds |

---

## Correct Flow

```
┌─────────────────────────────────────────────────────────────┐
│ ITERATION 1: Initial Strategy                               │
└─────────────────────────────────────────────────────────────┘

Strategy Agent
  ↓ Queries database: "No similar strategies tried yet"
  ↓ Generates: RSI < 20 entry, TrendStrength > 0.7 exit
  ↓
Code Generator → Python class
  ↓
Backtest → Results: 0 trades, no equity curve
  ↓
Filter → REJECTED (< 10 trades threshold)
  ↓
Critique Agent → Analyzes:
  {
    "root_cause": "entry_too_restrictive",
    "evidence": {
      "num_trades": 0,
      "entry_conditions": "RSI(14) < 20 AND TrendStrength > 0.7",
      "diagnosis": "RSI < 20 occurs only 4.6% of time, AND with TrendStrength > 0.7 is < 1%"
    },
    "recommendations": {
      "relax_rsi": "Try RSI < 40-45 (30th percentile)",
      "relax_trend": "Try TrendStrength > 0.5 or remove it",
      "strategy": "Use OR instead of AND for more signals"
    }
  }
  ↓
Database → Stores:
  - strategy_id: 1
  - status: 'rejected'
  - critique_analysis: {root_cause, recommendations}

┌─────────────────────────────────────────────────────────────┐
│ ITERATION 2: Refined Strategy (Strategy Agent Learns)       │
└─────────────────────────────────────────────────────────────┘

Strategy Agent
  ↓ Queries database:
  |   - get_failed_strategies(reason='entry_too_restrictive')
  |   - Finds: strategy_id=1 with critique analysis
  ↓ Reads critique:
  |   "RSI < 20 too restrictive, try 40-45"
  |   "TrendStrength > 0.7 too strict, try 0.5"
  ↓ LLM Decision:
  |   Context: "Previous strategy RSI < 20 got 0 trades"
  |   Critique: "Relax to RSI < 40-45"
  |   Decision: "Generate refinement with RSI < 44, TrendStrength > 0.5"
  ↓ Generates:
  {
    "strategy_type": "mean_reversion",
    "entry_conditions": {
      "logic": "(RSI(14) < 44 OR RSI(5) < 35) AND TrendStrength > 0.5",
      "rationale": "Learning from strategy_id=1 failure, relaxed thresholds"
    },
    "parent_id": 1,
    "generation": 1,
    "refinement_type": "learned_from_failure"
  }
  ↓
Backtest → Results: 18 trades, Sharpe 1.2
  ↓
Filter → APPROVED ✓
  ↓
Critique Agent → Analyzes:
  {
    "status": "success",
    "strengths": ["Good trade frequency", "Positive Sharpe"],
    "potential_improvements": ["Consider tighter stop loss for better DD"]
  }
  ↓
Database → Stores success
```

---

## Key Insight

**Strategy Agent is the "brain" - it makes ALL decisions about what strategies to generate.**

**Critique Agent is the "teacher" - it provides feedback but doesn't make decisions.**

This is analogous to:
- **Strategy Agent** = Student (decides what to study, generates answers)
- **Critique Agent** = Teacher (grades answers, explains mistakes)

The student (Strategy Agent) learns from the teacher's feedback, but the teacher doesn't do the student's homework.

---

## Implementation: Strategy Agent Modifications

### Add Method: `_gather_failure_context()`

```python
# src/agents/strategy_agent.py

def _gather_failure_context(self) -> Dict[str, Any]:
    """
    Query database for failed strategies and extract learnings.

    Returns:
        Dict with failure patterns and critique recommendations
    """
    # Get rejected strategies
    rejected = self.database.get_strategies_by_status('rejected', limit=50)

    # Get strategies with 0 trades
    zero_trades = self.database.get_strategies_with_zero_trades(limit=20)

    # Get strategies with low Sharpe (< 0.3)
    low_sharpe = self.database.get_strategies_below_sharpe(threshold=0.3, limit=20)

    # Get critique analysis for these failures
    failure_patterns = {
        'entry_too_restrictive': [],
        'exit_too_tight': [],
        'no_risk_management': [],
        'too_many_trades': []
    }

    for strategy in rejected + zero_trades + low_sharpe:
        # Get critique analysis from database
        critique = self.database.get_critique_analysis(strategy['id'])

        if critique:
            root_cause = critique.get('root_cause')
            if root_cause in failure_patterns:
                failure_patterns[root_cause].append({
                    'strategy_id': strategy['id'],
                    'entry_conditions': strategy['entry_rules'],
                    'exit_conditions': strategy['exit_rules'],
                    'recommendations': critique.get('recommendations', {})
                })

    return {
        'total_failures': len(rejected),
        'zero_trade_count': len(zero_trades),
        'failure_patterns': failure_patterns,
        'common_mistakes': self._identify_common_mistakes(failure_patterns)
    }

def _identify_common_mistakes(self, failure_patterns: Dict) -> List[str]:
    """
    Identify recurring mistakes across multiple failures.

    Example: If 10 strategies failed with "RSI < 25", learn to avoid it.
    """
    mistakes = []

    # Check for repeated parameter values in failures
    entry_restrictive = failure_patterns.get('entry_too_restrictive', [])

    if len(entry_restrictive) >= 3:
        # Extract common patterns
        rsi_thresholds = []
        for failure in entry_restrictive:
            # Parse entry conditions for RSI thresholds
            # This is simplified - actual implementation would parse the logic string
            if 'RSI' in failure['entry_conditions']:
                # Extract threshold (e.g., "RSI < 25" → 25)
                pass

        mistakes.append(f"RSI < 25 failed {len(entry_restrictive)} times - avoid extremely low thresholds")

    return mistakes
```

### Modify: `_gather_context()` to include failures

```python
def _gather_context(self) -> Dict[str, Any]:
    """Gather relevant context from database."""
    logger.info("Gathering context from database...")

    # ... existing code for successes ...

    # NEW: Add failure context
    failure_context = self._gather_failure_context()

    context = {
        # ... existing fields ...
        'failures': failure_context,  # NEW
        'learnings': self._extract_learnings(failure_context)  # NEW
    }

    return context

def _extract_learnings(self, failure_context: Dict) -> List[str]:
    """
    Convert failure patterns into actionable learnings.

    Examples:
    - "Avoid RSI < 25 - failed 10 times with 0 trades"
    - "TrendStrength > 0.7 too strict - relax to > 0.5"
    - "AND conditions too restrictive - prefer OR for more signals"
    """
    learnings = []

    patterns = failure_context['failure_patterns']

    # Entry too restrictive patterns
    for failure in patterns.get('entry_too_restrictive', []):
        recs = failure.get('recommendations', {})
        if 'relax_rsi' in recs:
            learnings.append(f"Learning: {recs['relax_rsi']}")

    # Common mistakes
    learnings.extend(failure_context.get('common_mistakes', []))

    return learnings
```

### Modify: `_build_exploration_prompt()` to include learnings

```python
def _build_exploration_prompt(self, context: Dict, models: List) -> str:
    """Build LLM prompt with successes AND failures."""

    prompt = f"""
    You are an autonomous quantitative strategy researcher.

    ## Historical Context

    Total strategies tested: {context['total_strategies']}

    ### Top Performers (Learn from successes):
    {self._format_top_performers(context['top_performers'])}

    ### Recent Failures (Learn from mistakes):  <<<< NEW
    Total failures: {context['failures']['total_failures']}
    Zero-trade strategies: {context['failures']['zero_trade_count']}

    ### Key Learnings:  <<<< NEW
    {chr(10).join(f"- {learning}" for learning in context['learnings'])}

    ### Failure Patterns:  <<<< NEW
    {self._format_failure_patterns(context['failures']['failure_patterns'])}

    ## Your Task

    Generate a NEW strategy that:
    1. Learns from past failures (avoid repeated mistakes)
    2. Explores underexplored areas
    3. Uses realistic thresholds based on learnings above

    If there are recent failures with specific critiques, you may generate
    a REFINEMENT that addresses the critique's recommendations.

    Examples:
    - If RSI < 20 failed 5 times → Try RSI < 40 (relaxed threshold)
    - If TrendStrength > 0.7 caused 0 trades → Try > 0.5 or remove it
    - If AND conditions too restrictive → Use OR logic

    Generate strategy specification:
    ...
    """

    return prompt
```

---

## Implementation: Critique Agent Modifications

### Critique Agent ONLY Analyzes, Does NOT Generate

```python
# src/critique/critique_agent.py

class CritiqueAgent:
    """
    Analyzes strategy performance and provides structured feedback.

    DOES:
    - Diagnose why strategies failed/succeeded
    - Provide improvement recommendations
    - Identify root causes

    DOES NOT:
    - Generate strategy specifications
    - Decide what to try next
    - Create strategy metadata
    """

    def analyze_strategy(self, strategy_id: int) -> Dict[str, Any]:
        """
        Analyze a strategy's performance.

        Returns:
            Structured analysis with root cause and recommendations
        """
        strategy = self.database.get_strategy(strategy_id)
        backtest = self.database.get_backtest_results(strategy_id)

        # Classify failure type
        failure_type = self._classify_failure(backtest)

        # Get LLM analysis
        analysis = self._get_llm_analysis(strategy, backtest, failure_type)

        # Store in database for Strategy Agent to query later
        self.database.store_critique_analysis(
            strategy_id=strategy_id,
            analysis=analysis
        )

        return analysis

    def _classify_failure(self, backtest: Dict) -> str:
        """
        Classify the type of failure.

        Returns:
            One of: entry_too_restrictive, exit_too_tight,
                    no_risk_management, too_many_trades, no_edge
        """
        metrics = backtest['metrics']

        if metrics['num_trades'] == 0:
            return 'entry_too_restrictive'
        elif metrics['num_trades'] < 5:
            return 'entry_still_too_restrictive'
        elif metrics['win_rate'] < 0.25:
            return 'exit_too_tight'
        elif metrics['max_drawdown'] > 0.35:
            return 'no_risk_management'
        elif metrics['num_trades'] > 200:
            return 'too_many_trades'
        elif metrics['sharpe_ratio'] < 0.3:
            return 'no_edge'
        else:
            return 'marginal_performance'

    def _get_llm_analysis(
        self,
        strategy: Dict,
        backtest: Dict,
        failure_type: str
    ) -> Dict[str, Any]:
        """
        Get LLM to provide detailed analysis and recommendations.

        Returns structured feedback for Strategy Agent to consume.
        """
        prompt = f"""
        Analyze this strategy's performance:

        Strategy: {strategy['strategy_type']}
        Entry: {strategy['entry_rules']}
        Exit: {strategy['exit_rules']}

        Results:
        - Trades: {backtest['metrics']['num_trades']}
        - Sharpe: {backtest['metrics']['sharpe_ratio']:.2f}
        - Win Rate: {backtest['metrics']['win_rate']:.1%}
        - Max DD: {backtest['metrics']['max_drawdown']:.1%}

        Failure Type: {failure_type}

        Provide:
        1. Root cause (why it failed)
        2. Specific parameter issues (e.g., "RSI < 20 too strict")
        3. Recommendations (e.g., "relax to RSI < 40-45")
        4. Alternative approaches

        Return as JSON:
        {{
          "root_cause": "...",
          "evidence": {{}},
          "recommendations": {{
            "parameter_adjustments": [],
            "logic_changes": [],
            "alternative_indicators": []
          }}
        }}
        """

        response = self.llm_client.complete(prompt, response_format="json")
        return json.loads(response)
```

---

## Database Schema Addition

Need to store critique analysis:

```python
# src/database/schema.py

class CritiqueAnalysis(Base):
    """Stores critique feedback for Strategy Agent to learn from"""
    __tablename__ = 'critique_analysis'

    id = Column(Integer, primary_key=True)
    strategy_id = Column(Integer, ForeignKey('strategies.id'), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Analysis results
    root_cause = Column(String)  # entry_too_restrictive, etc.
    evidence = Column(JSON)       # Why it failed
    recommendations = Column(JSON)  # How to improve

    # Link to strategy
    strategy = relationship("Strategy", back_populates="critique")
```

---

## Summary: Proper Roles

| Component | Role | Analogy |
|-----------|------|---------|
| **Strategy Agent** | Decision Maker | Student - decides what to study, generates answers |
| **Critique Agent** | Advisor | Teacher - grades work, explains mistakes |
| **Code Generator** | Implementer | Compiler - translates specs to executable code |
| **Filter** | Gatekeeper | Exam - pass/fail based on criteria |
| **Database** | Memory | Notebook - records everything for future reference |

---

## What Changes in the Roadmap

### Before (Flawed):
- Critique Agent generates refinements ❌

### After (Correct):
- Critique Agent only analyzes and stores feedback ✅
- Strategy Agent queries critique feedback and generates ALL strategies ✅
- Strategy Agent learns from both successes AND failures ✅

---

## Benefits of This Architecture

1. **Single Source of Truth**: Strategy Agent is THE decision maker
2. **Critique is Reusable**: Multiple iterations can learn from same critique
3. **Composable Learning**: Strategy Agent can combine learnings from 100 failed strategies
4. **Clear Boundaries**: Each component has one job
5. **Testable**: Can test Strategy Agent's learning independently from Critique quality
6. **Scalable**: Strategy Agent can query patterns across 1000s of failures

---

## Next Steps

1. **Enhance Strategy Agent**:
   - Add `_gather_failure_context()`
   - Modify prompts to include learnings
   - Query critique analysis from database

2. **Implement Critique Agent** (analysis only):
   - `analyze_strategy()` - root cause + recommendations
   - Store in database, don't generate strategy specs

3. **Add Database Table**:
   - `critique_analysis` table
   - Methods to query by root_cause, recommendations

4. **Test Learning Loop**:
   - Generate 10 strategies with RSI < 20
   - All fail with 0 trades
   - Strategy Agent sees pattern, avoids RSI < 20 in next 10
   - Success rate improves

---

**The Strategy Agent is the autonomous learner. The Critique Agent is just its feedback mechanism.**
