# ============================================================
# GLOBAL CONFIG
# ============================================================

DEBUG = False  # NEVER set True in competition submissions — causes Lambda timeouts

# Position limits per product (updated each round)
LIMITS: dict[str, int] = {
    "EMERALDS": 80,
    "TOMATOES": 80,
}

# Master switch: "aggressive" = wide market making, take spreads freely
#                "conservative" = only take near-certain arb, tight sizing
# Can be overridden per-product based on realised volatility regime.
MASTER_SWITCH: str = "aggressive"
