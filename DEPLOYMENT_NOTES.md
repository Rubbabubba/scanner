# patch-035-contract-safe-7-symbol-lock-rb1-preservation

Changes:
- Lock scanner forced emit pilot to the 7 resolvable symbols:
  BTC/USD,ETH/USD,SOL/USD,ADA/USD,LINK/USD,AVAX/USD,DOT/USD
- Contract any prior 10-symbol pilot request back to the safe 7-symbol set.
- Lower scanner TOP_N default from 10 to 7.
- Preserve ranking, spread filtering, coordination, and emission controls.
