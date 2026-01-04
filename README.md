# KingShot Gear Optimizer

A gear upgrade optimizer for KingShot that uses multi-stage greedy algorithms to maximize stat efficiency across multiple game scenarios.

## Overview

The optimizer operates in two phases:
1. **Forgehammer Optimization**: Determines the most efficient mastery upgrades based on your forgehammer budget and investment priorities
2. **XP Allocation**: Greedily applies available gear XP to maximize stat gains per XP spent

## Key Features

- **Multi-Scenario Scoring**: Evaluates gear value across different game modes (Conquest, Expedition) with customizable scenario weights
- **Hero-Based Strategy**: Configure per-hero, per-slot investment multipliers (prioritize, normal, or ban specific gear pieces)
- **Intelligent Forgehammer Planning**: 
  - Gold-tier filtering with explicit investment whitelist
  - Efficiency thresholds (absolute or relative to peak)
  - Mission-aware advice for Alliance Mobilization quests
- **Flexible Stat Profiles**: Per-hero stat weights with scenario-specific overrides
- **Loadout Management**: Track hero-gear assignments, fork loadouts, swap gear between heroes

### Requirements

- Python 3.7+
- Standard library only (no external dependencies)

### Usage

#### Basic

1. Modify `create_inventory` so it matches your available gears. For example `make("marlin_helm", "Marlin Gold Arc Helmet", 76, 'Gold', 'Gold_Arc_HB', 1.50),` means archer helmet, labeled as Marlin's, level 76 and Mastery level 5. If you have level 80, change 76 to 80. If you have Mastery 7, change 1.50 to 1.70.

2. Update the value of `hammer_budget`. If you have 69 Forgehammers and you are ready to use them, write 69.

3. Update the value of `xp_budget` for the amount of free gear XP you have in your inventory. If you are unsure, enter a number like 3000 to see a few upgrades at a time.

4. Run: `python KingShot_gear_optimizer.py`

#### Advanced

1. Clone the repository
2. Edit `KingShot_gear_optimizer.py` to configure:
   - Your gear inventory (stats, levels, tiers)
   - Hero stat weight profiles
   - Forgehammer strategy (which heroes/slots to prioritize)
   - Game scenarios and their importance weights
   - Available forgehammer and XP budgets
3. Run: `python KingShot_gear_optimizer.py`

The optimizer will output:
- Eligibility report showing which gear can receive forgehammer upgrades
- Strategic roadmap with recommended upgrade sequence
- Final gear upgrade plan with XP allocation by hero

### Configuration Example

```python
# Define your stat priorities per hero. They serve as the default weights that can be tweaked downstream
baselineprofiles = {
    'Marlin': (8.0, 0.7, 0.7, 2.0, 0.7, 0.7, 5.0, 2.0),  # High value in offensive gears
    'Zoe': (0.1, 1.0, 1.0, 0.05, 1.0, 1.0, 2.0, 6.0),     # High value in def gears, some value in offensive gears
}

# Set forgehammer investment strategy (Helmet, Chest, Gloves, Boots)
FORGESTRATEGY = {
    'Marlin': (2.0, 1.0, 1.0, 2.0),  # Increased priority on helmet/boots
    "Zoe": (0.5, 1.0, 1.0, 0.5),     # Reduced priority on helmet/boots
}

# Define game scenarios with weights
scenarios = [
    (conquest_loadout, GameMode.CONQUEST, 2.0),    # High importance
    (expedition_loadout, GameMode.EXPEDITION, 1.0), # Standard importance
]
```

## Roadmap

- [ ] Configuration file support (JSON/YAML)
- [ ] Interactive CLI with prompts
- [ ] UI for easier gear input and visualization
- [ ] Import/export functionality for sharing builds

## Technical Details

### Algorithm

**Forgehammer Phase**: 
- Calculates efficiency as `(stat_gain × investment_multiplier) / hammer_cost`
- Uses max-heap priority queue to select optimal upgrades
- Respects tier restrictions and explicit investment multipliers

**XP Phase**:
- Greedy selection based on `weighted_stats_per_level / xp_to_apply`
- Applies XP in multiples of 10 (game rule)
- Tracks partial XP for next level

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Author

**ecpgieicg** - [GitHub Profile](https://github.com/ecpgieicg)

## Acknowledgments

Built for the KingShot community. Special thanks to alliance members who helped test and refine the optimization logic.
