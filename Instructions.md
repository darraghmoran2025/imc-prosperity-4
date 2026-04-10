# MISSION
Act as a Lead Quant Researcher. Your goal is to build a high-performance Python trading repository for the IMC Prosperity 4 challenge.

# RESEARCH & SOURCING
1. Use the browser tool to search GitHub for "IMC Prosperity 3 winners" and "IMC Prosperity 2 winners". 
2. Specifically look for repos from top teams like "Frankfurt Hedgehogs", "CMU Physics", and "Alpha Animals". 
3. Analyze their strategies for:
   - Market Making (Amethysts/Starfruit equivalent)
   - Pair Trading / Statistical Arbitrage (Orchids/Basket equivalent)
   - Options Pricing (Black-Scholes implementations from Round 4)
4. Extract successful logic for: position management, inventory skewing, and risk limits.

# IMPLEMENTATION SPECIFICATIONS
1. **Repository Structure:** Create a clean Python repo that complies with IMC Prosperity 4 specs. Use a build script if necessary to bundle modular logic into a single `solution.py`.
2. **Backtesting:** Use the latest community backtester (search for 'prosperity4bt'). Do not build a custom backtester; rely on the community standard to ensure exchange-accurate behavior.
3. **Execution & Limits:** - Optimize for low latency and minimal computational overhead.
   - Implement a strict `DEBUG = False` flag. 
   - **Crucial:** Minimize logging to avoid AWS Lambda timeouts and "Verbose Logging" disqualifications.
4. **Strategy Innovation:** - Look for "Information Advantage" patterns (shadowing "insider" bots) common in Round 5.
   - Implement a "Master Switch" to toggle between aggressive market making and conservative arbitrage based on market volatility.
   - Design for robustness: include safety checks for position limits and accidental "wash trading."

# OPERATIONAL RULES
1. **No Attribution:** Do not list "Claude" or "AI" as a contributor in any files, metadata, or the README.
2. **Autonomy with Check-ins:** You have full creative freedom to develop novel trading signals or logic if you believe they will outperform previous winners. 
3. **If Unsure:** STOP and ask me for clarification if you encounter ambiguous competition rules or conflicting data from your research.
4. **Verification:** You MUST verify your code against the competition's technical constraints (e.g., specific Python version, library restrictions).

# FIRST STEP
Research the current Round 1 products for Prosperity 4 (2026) and propose a project structure and a research summary of the top 3 strategies you've found.