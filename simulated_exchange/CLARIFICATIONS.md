# SimulatedExchange Implementation Clarifications

Please answer the following questions to guide the initial implementation:

## 1. Project Scope & Phase

**Q1.1**: Which phase should we target for the initial implementation?
- [ ] Phase 1 only (Core MVP - basic backtesting with market orders)
- [ ] Phase 1 + Phase 2 (Enhanced backtesting with limit orders)
- [ ] All phases up to Phase 3 (including live paper trading)

**Your answer**:
All phases
---

## 2. Data & Testing

**Q2.1**: Do you have historical price data files ready?
- [ ] Yes, I have data files ready (please specify format and location below)
- [ ] No, please create sample/mock data for testing
- [ ] Please implement a data downloader to fetch from an exchange

**Your answer**:
Please implement a data downloader to fetch from an exchange

**If you have data, please specify**:
- File path(s):
- Format: [ ] CSV [ ] Parquet [ ] Other: ___________
- Symbols included:
- Date range:
- Timeframe:

---

## 3. Feature Prioritization

**Q3.1**: For Phase 1 MVP, should we implement:
- [ ] All slippage models (fixed, volume-based, hybrid)
- [ ] Just fixed slippage (simplest)
- [ ] Just hybrid slippage (most realistic)

**Your answer**:
Fixed

**Q3.2**: Should limit orders be included in Phase 1?
- [ ] Yes, include limit orders in Phase 1
- [ ] No, market orders only for Phase 1 (add limit orders in Phase 2)

**Your answer**:
Yes

**Q3.3**: Should risk management features (section 3.3 - position limits, max leverage) be included in Phase 1?
- [ ] Yes, include basic risk management from the start
- [ ] No, defer to a later phase
- [ ] Implement as optional configuration only

**Your answer**:
No
---

## 4. Live Trading & Exchange Integration

**Q4.1**: For LivePriceFeed implementation, which exchange should we prioritize?
- [ ] Hyperliquid
- [ ] Binance
- [ ] Both equally
- [ ] Skip for now (focus on backtesting only)

**Your answer**:
Both equally

**Q4.2**: Do you have API keys for the exchange?
- [ ] Yes, I have API keys ready
- [ ] No, but I can get them
- [ ] Not needed yet (testnet only)
- [ ] Skip live trading for now

**Your answer**:
No, but I can get them
---

## 5. Technical Preferences

**Q5.1**: Testing framework preference?
- [ ] pytest (recommended)
- [ ] unittest
- [ ] No preference

**Your answer**:
pytest


**Q5.2**: Should we implement logging from the start?
- [ ] Yes, comprehensive logging from day 1
- [ ] Basic logging only (errors and warnings)
- [ ] Add logging in Phase 4

**Your answer**:
Basic logging only

**Q5.3**: Configuration approach:
- [ ] Python dict/class-based configuration (as shown in section 11)
- [ ] YAML/JSON configuration files
- [ ] Both (dict for code, with ability to load from files)

**Your answer**:
Both

---

## 6. Development Approach

**Q6.1**: Development strategy preference:
- [ ] Build everything comprehensively, even if it takes longer
- [ ] Build minimal viable version quickly, then iterate
- [ ] Build feature by feature with testing at each step

**Your answer**:
Build minimal viable version quickly, then iterate

**Q6.2**: Should we create the full file structure (Appendix A) from the start?
- [ ] Yes, create complete structure with placeholder files
- [ ] No, create files only as needed
- [ ] Create core structure, skip docs/examples for now

**Your answer**:
No, create files only as needed
---

## 7. Additional Requirements

**Q7.1**: Are there any specific features or modifications you want that differ from the requirements doc?

**Your answer**:
No

**Q7.2**: Are there any features in the requirements doc you want to explicitly skip or defer?

**Your answer**:
No

**Q7.3**: Do you plan to integrate with a specific LLM framework or API?
- [ ] OpenAI API
- [ ] Anthropic Claude API
- [ ] Local LLM
- [ ] Not yet decided / will add later
- [ ] Other: ___________

**Your answer**:
OpenAI Azure/ OpenAI / Deepseek 
Might user pure llm or langgraph or openai agents sdk
---

## 8. Success Criteria

**Q8.1**: What would you consider a successful initial version? (Check all that apply)
- [ ] Can backtest a simple buy-and-hold strategy
- [ ] Can backtest with multiple orders and position tracking
- [ ] Can calculate accurate PnL and basic metrics
- [ ] Can generate equity curve and trade history
- [ ] Can run with real historical data
- [ ] Has passing unit tests
- [ ] Has example scripts that work

**Your answer**:
Can backtest a simple buy-and-hold strategy
Can backtest with multiple orders and position tracking
Can calculate accurate PnL and basic metrics
Can generate equity curve and trade history
Can run with real historical data

---

## 9. Timeline & Resources

**Q9.1**: Is there a timeline or deadline for this project?

**Your answer**:
No

**Q9.2**: Will you be testing/reviewing as we build, or should I complete a full section before showing you?

**Your answer**:
Full section

---

Please fill in your answers and let me know when you're ready for me to start implementation!
