"""
Gear Upgrade Optimizer v0.10 (Loadout Manager, Refactored)

==================================================
Three-tier XP cost system:
Level 0-29: cost = 10 + 5×level
Level 30-38: cost = 15×level - 280
Level 39+:  cost = 20×level - 510
==================================================
Forgehammer (Mastery) Cost system:
Cost  = (current_mastery_level + 1) * 10
Bonus = +0.1 (10%) per mastery level
==================================================
Potential to-dos
- take a configuration file
- a mode that continue to upgrade upon demand -- with the xp availability being very rough range
"""

import math
import heapq
from collections import defaultdict
from typing import Dict, List, Tuple, NamedTuple, Optional, Any

# --- CONFIGURATION: GLOBAL CONSTANTS ---

# Feature: Global Max Levels by Tier (Gold/Purple/Blue)
MAX_LEVEL_BY_TIER = {
    'Gold': 100,
    'Purple': 80,
    'Blue': 60,
}

# --- GAME MODES: WHICH STATS ARE ACTIVE PER SCENARIO ---


class GameMode:
    """
    Defines which stats are relevant for a specific game scenario.

    GameMode.CONQUEST:
      Uses only Hero/Escort stats (no Expedition stats).

    GameMode.EXPEDITION:
      Uses only Expedition stats (Lethality/Health).
    """

    CONQUEST = {
        'hero_atk', 'hero_def', 'hero_health',
        'escort_atk', 'escort_def', 'escort_health',
    }

    EXPEDITION = {
        'exped_lethality', 'exped_health',
    }


# --- STAT DATA STRUCTURES ---


class GearStats(NamedTuple):
    """
    Stats provided by gear per level.

    Fields (per level):
      hero_atk        : Hero Attack
      hero_def        : Hero Defense
      hero_health     : Hero HP
      escort_atk      : Escort Attack
      escort_def      : Escort Defense
      escort_health   : Escort HP
      exped_lethality : Expedition Lethality
      exped_health    : Expedition HP
    """
    hero_atk: float
    hero_def: float
    hero_health: float
    escort_atk: float
    escort_def: float
    escort_health: float
    exped_lethality: float
    exped_health: float

    def calculate_weighted_value(self, profile: "StatProfile", active_stats: set) -> float:
        """
        Calculate weighted stat value for this gear under a given profile,
        using only stats relevant to the active GameMode.

        This is the core per-scenario stat-scoring function:
          base_score(gear, scenario) = sum_over_active_stats( stat_value * hero_weight )
        """
        score = 0.0
        for stat_key in active_stats:
            val = getattr(self, stat_key, 0.0)
            weight = profile.get_weight(stat_key)
            score += val * weight
        return score


class StatProfile:
    """
    Maps a user's configuration (Tuple or Dict) to specific named stat weights.

    The underlying engine always works with the full 8 stat keys below.
    UI/config can supply them as:
      - 8-float tuple in fixed order, or
      - dict keyed by full stat names.

    ORDERED_KEYS (full names):
      hero_atk, hero_def, hero_health,
      escort_atk, escort_def, escort_health,
      exped_lethality, exped_health
    """

    ORDERED_KEYS = (
        'hero_atk', 'hero_def', 'hero_health',
        'escort_atk', 'escort_def', 'escort_health',
        'exped_lethality', 'exped_health',
    )

    def __init__(self, data: Any):
        # Internal representation: dict[str, float]
        self.weights: Dict[str, float] = {}

        if isinstance(data, (tuple, list)):
            if len(data) != 8:
                raise ValueError(f"Tuple profile must have exactly 8 values. Got {len(data)}.")
            self.weights = dict(zip(self.ORDERED_KEYS, data))
        elif isinstance(data, dict):
            # Allow partial dicts: unspecified keys default to 0.0 (caller may merge a baseline first).
            for key in self.ORDERED_KEYS:
                self.weights[key] = float(data.get(key, 0.0))
        else:
            raise TypeError("Profile must be an 8-float tuple/list or a dictionary keyed by stat name.")

    def get_weight(self, stat_name: str) -> float:
        return self.weights.get(stat_name, 0.0)


# --- STAT / XP / HAMMER FUNCTIONS ---


def xp_cost_for_level(level: int) -> int:
    if level < 30:
        return 10 + 5 * level
    elif level < 40:
        return 10 * level - 130
    elif level < 60:
        return 20 * level - 510
    elif level < 70:
        return 30 * level - 1120
    elif level < 80:
        return 40 * level - 1810
    else:
        return 50 * level - 2600


def hammer_cost_for_mastery(mastery_level: int) -> int:
    """
    Calculate Forgehammers needed to upgrade from mastery_level to mastery_level+1
    Formula: (current_mastery_level + 1) * 10
    """
    return (mastery_level + 1) * 10


# --- ALLIANCE MOBILIZATION CONFIG (UNCHANGED, FOR REFERENCE) ---

ALLIANCE_MOBILIZATION_QUESTS = {
    'Blue': 10,
    'Purple': 20,
    'Gold': 35,
}


# --- CORE GEAR OBJECT ---


class GearPiece:
    """Represents a single gear piece with detailed stat tracking."""

    def __init__(self, id: str, name: str, level: int, tier: str, stats: GearStats,
                 gear_level_bonus: float = 1.0):
        self.id = id
        self.name = name
        self.level = level
        self.tier = tier  # 'Gold', 'Purple', 'Blue'
        self.stats = stats
        self.partial_xp = 0

        # Mastery / Forgehammer tracking
        self.gear_level_bonus = gear_level_bonus
        # Convert bonus (e.g. 1.2) back to integer level (2)
        # Rounding to handle potential floating point drift
        self.mastery_level = int(round((gear_level_bonus - 1.0) * 10))

        # Max Level determined by Tier
        self.max_level = MAX_LEVEL_BY_TIER.get(tier, 60)

        # Calculated metrics (set by scoring)
        self.composite_weight_score = 0.0  # Utility across all scenarios
        self.hammer_priority_score = 0.0   # Legacy metric; not used by new hammer engine

        # NEW: Explicit Forgehammer investment multiplier (0.0 = BANNED)
        self.hammer_multiplier = 0.0

        # Initialize default weighted stats (will be overwritten by scoring)
        self.weighted_stats_per_level = 0.0

    def set_composite_metrics(self, weight_score: float, hammer_prio: float):
        """
        Assign final calculated utility of this piece.

        For the new engine:
          - weight_score is the composite XP value (sum over scenarios).
          - hammer_prio is kept only for legacy compatibility; new hammer logic
            uses hammer_multiplier instead.
        """
        self.composite_weight_score = weight_score
        self.hammer_priority_score = hammer_prio  # kept but unused by new hammer optimizer
        self.update_weighted_stats()

    def update_weighted_stats(self):
        """Recalculate weighted stats per level based on current bonus."""
        self.weighted_stats_per_level = self.composite_weight_score * self.gear_level_bonus

    # --- XP Methods ---

    def cost_for_next_level(self) -> int:
        return xp_cost_for_level(self.level)

    def xp_needed_for_next_level(self) -> int:
        return self.cost_for_next_level() - self.partial_xp

    def xp_to_apply(self) -> int:
        # XP is always applied in multiples of 10 (game rule)
        return math.ceil(self.xp_needed_for_next_level() / 10) * 10

    def efficiency(self) -> float:
        xp = self.xp_to_apply()
        if xp <= 0:
            return 0.0
        return self.weighted_stats_per_level / xp

    def can_upgrade(self, available_xp: int) -> bool:
        return (self.level < self.max_level and self.xp_to_apply() <= available_xp)

    def apply_xp(self, xp_amount: int) -> Tuple[bool, int]:
        """
        Apply XP and return (leveled_up, leftover_xp).
        leftover_xp is kept as partial_xp for the next level.
        """
        self.partial_xp += xp_amount
        cost_needed = self.cost_for_next_level()
        if self.partial_xp >= cost_needed:
            leftover = self.partial_xp - cost_needed
            self.level += 1
            self.partial_xp = leftover
            return True, leftover
        return False, 0

    # --- Forgehammer Methods ---

    def hammer_cost_next(self) -> int:
        return hammer_cost_for_mastery(self.mastery_level)

    def hammer_efficiency(self) -> float:
        """
        LEGACY helper; not used by the new ForgehammerOptimizer.

        New hammer logic uses:
          gain = composite_weight_score * 0.10 * hammer_multiplier
          cost = hammer_cost_next()
          eff  = gain / cost
        """
        if self.hammer_multiplier <= 0.0:
            return 0.0
        gain = self.composite_weight_score * 0.10 * self.hammer_multiplier
        cost = self.hammer_cost_next()
        return gain / cost

    def apply_hammer(self) -> bool:
        """Apply one mastery level and update bonus."""
        self.mastery_level += 1
        self.gear_level_bonus += 0.10
        self.update_weighted_stats()
        return True


# --- GENERIC LOADOUT STRUCTURE ---


class Loadout:
    """
    A named mapping of Heroes to Gear slots.

    Internal structure:
      _assignments: { 'HeroName': { 'SlotName': 'GearID' } }
      _gear_lookup: { 'GearID': 'HeroName' }
    """

    def __init__(self, name: str):
        self.name = name
        self._assignments: Dict[str, Dict[str, str]] = {}
        self._gear_lookup: Dict[str, str] = {}

    def assign(self, hero_name: str, slot: str, gear_id: str):
        if hero_name not in self._assignments:
            self._assignments[hero_name] = {}
        self._assignments[hero_name][slot] = gear_id
        self._gear_lookup[gear_id] = hero_name

    def get_hero_gear(self, hero_name: str) -> Optional[Dict[str, str]]:
        """Return the slot->gear_id dict for a hero (or None if missing)."""
        return self._assignments.get(hero_name)

    def get_hero_for_gear(self, gear_id: str) -> Optional[str]:
        """Return the hero currently wearing this gear in this loadout."""
        return self._gear_lookup.get(gear_id)

    def validate_integrity(self) -> Tuple[bool, List[str]]:
        """
        Check that no gear ID is assigned to more than one hero in this loadout.
        Returns (is_valid, errors).
        """
        seen: Dict[str, str] = {}
        errors: List[str] = []
        for hero, slots in self._assignments.items():
            for slot, gid in slots.items():
                if gid in seen:
                    errors.append(
                        f"Gear {gid} assigned to both {seen[gid]} and {hero} in {self.name}"
                    )
                seen[gid] = hero
        return len(errors) == 0, errors

    def clone(self, new_name: str) -> "Loadout":
        """Deep copy this loadout under a new name."""
        import copy
        new_obj = Loadout(new_name)
        new_obj._assignments = copy.deepcopy(self._assignments)
        new_obj._gear_lookup = copy.deepcopy(self._gear_lookup)
        return new_obj


class LoadoutManager:
    """
    Registry of named Loadouts.

    Responsibilities here are *only*:
      - Create/load/fork loadouts.
      - Swap heroes/slots within a single loadout.
      - Validate integrity.

    All scoring logic is handled by GameScoringEngine.
    """

    def __init__(self):
        self.loadouts: Dict[str, Loadout] = {}

    def create(self, name: str) -> Loadout:
        self.loadouts[name] = Loadout(name)
        return self.loadouts[name]

    def get(self, name: str) -> Loadout:
        return self.loadouts[name]

    def fork(self, src: str, dst: str) -> Loadout:
        if src not in self.loadouts:
            raise ValueError(f"Unknown loadout: {src}")
        self.loadouts[dst] = self.loadouts[src].clone(dst)
        return self.loadouts[dst]

    def swap_heroes(self, loadout_name: str, hero_a: str, hero_b: str):
        lo = self.loadouts[loadout_name]
        gear_a = lo._assignments.get(hero_a, {})
        gear_b = lo._assignments.get(hero_b, {})

        # Clear old lookups
        for gid in gear_a.values():
            lo._gear_lookup.pop(gid, None)
        for gid in gear_b.values():
            lo._gear_lookup.pop(gid, None)

        # Swap assignments
        lo._assignments[hero_a], lo._assignments[hero_b] = gear_b, gear_a

        # Rebuild lookups
        for gid in gear_b.values():
            lo._gear_lookup[gid] = hero_a
        for gid in gear_a.values():
            lo._gear_lookup[gid] = hero_b

    def swap_slots(self, loadout_name: str, hero_a: str, hero_b: str, slots: List[str]):
        """
        Precise Swap: Swaps only specific slots between two heroes.
        """
        lo = self.loadouts[loadout_name]
        gear_a = lo._assignments.get(hero_a, {})
        gear_b = lo._assignments.get(hero_b, {})

        for slot in slots:
            id_a = gear_a.get(slot)
            id_b = gear_b.get(slot)

            # Update assignments
            if id_b:
                gear_a[slot] = id_b
            if id_a:
                gear_b[slot] = id_a

            # Update reverse lookup
            if id_b:
                lo._gear_lookup[id_b] = hero_a
            if id_a:
                lo._gear_lookup[id_a] = hero_b


# --- COMPONENT 2: GAME SCORING ENGINE (Separated from Loadout) ---


class GameScoringEngine:
    """
    Calculates composite utility for all gear based on provided scenarios.

    Conceptually:
      For each scenario s:
        - game_mode_s defines which stats are active.
        - loadout_s defines which hero wears which gear.
        - hero profiles (baseline ⊕ optional per-scenario overrides) define stat weights.
        - scenario_weight_s defines scenario importance.

      GearValue(gear) =
          sum_over_s( scenario_weight_s * scenario_utility_s(gear) )

      where:
        scenario_utility_s(gear) =
            sum_over_active_stats( stat_value(gear) * hero_weight_s(hero_wearing_gear) )

    Args:
        inventory_registry: Dict[gear_id, GearPiece]
        hero_profiles: Dict[HeroName, StatProfile]
            Baseline per-hero stat weights (scenario-agnostic).
        scenario_profile_overrides: Optional[
            Dict[int, Dict[HeroName, Dict[stat_name, float]]]
        ]
            Per-scenario, per-hero, per-stat overrides. Only specified
            stats are changed; others inherit baseline.
    """

    def __init__(
        self,
        inventory_registry: Dict[str, GearPiece],
        hero_profiles: Dict[str, StatProfile],
        scenario_profile_overrides: Optional[
            Dict[int, Dict[str, Dict[str, float]]]
        ] = None,
    ):
        self.inventory = inventory_registry
        # Treat hero_profiles as baseline profiles
        self.baseline_profiles: Dict[str, StatProfile] = hero_profiles
        # Optional per-scenario overrides
        self.scenario_profile_overrides: Dict[int, Dict[str, Dict[str, float]]] = (
            scenario_profile_overrides or {}
        )

    def _get_effective_profile(self, hero: str, scenario_index: int) -> Optional[StatProfile]:
        """
        Merge baseline profile with per-scenario overrides (if any)
        for this hero and scenario.
        """
        base = self.baseline_profiles.get(hero)
        if base is None:
            return None

        overrides_for_s = self.scenario_profile_overrides.get(scenario_index)
        if not overrides_for_s or hero not in overrides_for_s:
            return base

        override_dict = overrides_for_s[hero]
        # Merge: start from baseline weights, override specific keys
        eff_dict = dict(base.weights)
        for k, v in override_dict.items():
            eff_dict[k] = float(v)
        return StatProfile(eff_dict)

    def calculate_utilities(self, scenarios: List[Tuple[Loadout, set, Optional[float]]]):
        """
        Calculates composite_weight_score for all gear pieces from a list of scenarios.

        scenarios:
          - Format A (no explicit weights):
                [(LoadoutObject, GameModeSet), ...]
            All scenarios are treated as equal weight (1.0).

          - Format B (explicit weights):
                [(LoadoutObject, GameModeSet, scenario_weight), ...]
            ALL scenarios must include a weight; otherwise an error is raised.

        This method enforces:
          - Either ALL scenarios have weights (length 3) or NONE do.
          - In the NONE case, scenario_weight is injected as 1.0 for each.
        """

        # 1. Validate and Normalize Weights
        has_weight = [len(s) == 3 for s in scenarios]
        normalized: List[Tuple[Loadout, set, float]] = []

        if all(has_weight):
            # All tuples already carry (loadout, mode, scenario_weight)
            normalized = [(s[0], s[1], float(s[2])) for s in scenarios]
        elif not any(has_weight):
            # No explicit weights: treat all scenarios as weight = 1.0
            normalized = [(s[0], s[1], 1.0) for s in scenarios]
        else:
            # Mixed case is ambiguous and disallowed
            raise ValueError("Ambiguous Config: Provide scenario weights for ALL scenarios or NONE.")

        # 2. Reset Scores (hammer_prio kept as legacy; not used by new hammer engine)
        for piece in self.inventory.values():
            piece.set_composite_metrics(0.0, piece.hammer_priority_score)

        # 3. Calculation Loop (sum over scenarios)
        for scenario_index, (loadout, active_stats, scenario_weight) in enumerate(normalized):
            for gear_id, piece in self.inventory.items():
                hero_name = loadout.get_hero_for_gear(gear_id)
                if not hero_name:
                    continue

                profile = self._get_effective_profile(hero_name, scenario_index)
                if profile is None:
                    continue

                # A. Calculate Stat Score for this scenario
                base_score = piece.stats.calculate_weighted_value(profile, active_stats)

                # B. Commit to Gear (sum of scenario contributions)
                current_score = piece.composite_weight_score
                piece.set_composite_metrics(
                    current_score + (base_score * scenario_weight),
                    piece.hammer_priority_score,  # unchanged legacy field
                )


# --- FORGEHAMMER STRATEGY MAPPING (UI -> GEAR) ---


def apply_forge_strategy(
    inventory: Dict[str, GearPiece],
    canonical_loadout: Loadout,
    strategy_map: Dict[str, Tuple[float, float, float, float]],
):
    """
    Applies Hero-based Forge strategy to specific Gear IDs using a canonical loadout.

    Args:
        inventory:
            Dict[id, GearPiece]
        canonical_loadout:
            Loadout object (The reference for who owns what FOR FORGEHAMMER ONLY).
            Other scenarios/loadouts DO NOT affect hammer config.
        strategy_map:
            Dict[HeroName, Tuple(Helm, Chest, Glove, Boot)]

            Tuple entries are per-slot investment multipliers:
              0.0 = BANNED (strict safety)
              1.0 = Normal investment
              2.0 = High priority investment
            Slot order is fixed:
              (Helmet, Chest, Gloves, Boots)

    Behavior:
        - Resets ALL hammer_multiplier to 0.0 (strict whitelist).
        - For each hero in strategy_map:
            - Looks up that hero's slots in the canonical_loadout.
            - Applies the tuple multipliers to the corresponding GearPiece.
        - Gears not found in canonical_loadout or strategy_map remain banned.
    """
    # 1. Reset all to 0.0 (Strict Whitelist Approach)
    for piece in inventory.values():
        piece.hammer_multiplier = 0.0

    # 2. Apply Strategy
    slot_names = ["Helmet", "Chest", "Gloves", "Boots"]

    for hero, multipliers in strategy_map.items():
        # Get assignments from the canonical loadout
        # Expecting assignments format: {'Helmet': 'id', ...}
        assignments = canonical_loadout.get_hero_gear(hero)
        if not assignments:
            continue

        for idx, slot_name in enumerate(slot_names):
            if idx >= len(multipliers):
                break
            gear_id = assignments.get(slot_name)
            if gear_id and gear_id in inventory:
                inventory[gear_id].hammer_multiplier = float(multipliers[idx])


# --- FORGEHAMMER OPTIMIZER (NEW ENGINE, MULTIPLIER-DRIVEN) ---


class ForgehammerOptimizer:
    """
    Optimizes mastery levels using a greedy algorithm with explicit
    investment multipliers and strict gear-level whitelist.

    Core efficiency formula:
      gain = composite_weight_score * 0.10 * hammer_multiplier
      cost = hammer_cost_for_mastery(current_mastery)
      eff  = gain / cost

    Only gears that satisfy:
      - tier == 'Gold'
      - hammer_multiplier > 0.0

    are considered eligible for investment.
    """

    def __init__(
        self,
        gear_pieces: List[GearPiece],
        hammer_budget: int,
        min_efficiency: float = 0.0,
        max_mastery_level: Optional[int] = None,
    ):
        self.gear_pieces = gear_pieces
        self.hammer_budget = hammer_budget
        self.min_efficiency = min_efficiency
        self.max_mastery_level = max_mastery_level

    def calculate_peak_efficiency(self) -> float:
        """
        Calculates theoretical max efficiency for relative thresholding.
        Only considers eligible (tier Gold, multiplier > 0) items.
        """
        peak = 0.0
        for piece in self.gear_pieces:
            if piece.tier != 'Gold':
                continue
            if piece.hammer_multiplier <= 0.0:
                continue
            gain = (piece.composite_weight_score * 0.10) * piece.hammer_multiplier
            cost = piece.hammer_cost_next()
            eff = gain / cost
            if eff > peak:
                peak = eff
        return peak

    def _print_eligibility_report(self) -> int:
        """Prints a verbose summary of what gears are allowed for investment."""
        print(f"\n{'-'*95}")
        print(f"FORGEHAMMER ELIGIBILITY INSPECTION")
        print(f"{'-'*95}")
        print(f"{'Gear Name':<35} {'Tier':<8} {'Inv. Mult':<10} {'Base Score':<10} {'Status'}")
        print(f"{'-'*95}")

        eligible_count = 0
        # Sort: Gold first, then by multiplier desc
        sorted_pieces = sorted(
            self.gear_pieces,
            key=lambda p: (p.tier != 'Gold', -p.hammer_multiplier),
        )

        for p in sorted_pieces:
            # Show only Gold or explicitly configured items
            if p.tier != 'Gold' and p.hammer_multiplier <= 0:
                continue

            if p.tier != 'Gold':
                status = "SKIP (Tier)"
            elif p.hammer_multiplier <= 0:
                status = "BANNED (0.0)"
            else:
                status = "ELIGIBLE"
                eligible_count += 1

            print(
                f"{p.name:<35} {p.tier:<8} "
                f"{p.hammer_multiplier:<10.1f} {p.composite_weight_score:<10.1f} {status}"
            )

        print(f"{'-'*95}")
        return eligible_count

    def generate_strategic_roadmap(self):
        """
        Simulates the upgrade path based on efficiency and explicit strategy.

        Uses a max-heap (priority queue) on efficiency. At each step:
          - Pops best-efficiency eligible upgrade.
          - Spends cost if within budget.
          - Prints mission advice keyed to cost brackets.
          - Increments mastery and recomputes efficiency for that piece.
        """
        print(f"\n{'='*95}")
        print(f"STRATEGIC HAMMER ROADMAP (Simulation)")
        print(f"{'='*95}")

        # --- VERBOSE INSPECTION ---
        eligible_count = self._print_eligibility_report()

        print(f"\nBudget: {self.hammer_budget} | Min Eff: {self.min_efficiency:.2f}")
        print(f"Eligible for Investment: {eligible_count} items")

        if eligible_count == 0:
            print(f"\n[WARNING] NO GEARS ARE ELIGIBLE FOR INVESTMENT.")
            print(" - All gears have hammer_multiplier = 0.0 (or are not Gold).")
            print(" - Ensure 'apply_forge_strategy' was called with a valid strategy.")
            print(f"{'='*95}\n")
            return

        # --- INITIALIZATION ---
        pq: List[Tuple[float, int, int]] = []
        sim_state: Dict[int, Dict[str, float]] = {}

        for i, piece in enumerate(self.gear_pieces):
            # 1. TIER FILTER (Gold Only)
            if piece.tier != 'Gold':
                continue

            # 2. PERMISSION FILTER (Explicit Multiplier)
            if piece.hammer_multiplier <= 0.0:
                continue

            sim_state[i] = {
                'mastery': piece.mastery_level,
                'bonus': piece.gear_level_bonus,
                'base_score': piece.composite_weight_score,
                'multiplier': piece.hammer_multiplier,
            }

            cost = hammer_cost_for_mastery(piece.mastery_level)
            gain = (sim_state[i]['base_score'] * 0.10) * sim_state[i]['multiplier']
            eff = gain / cost

            if eff >= self.min_efficiency:
                if self.max_mastery_level is None or piece.mastery_level < self.max_mastery_level:
                    heapq.heappush(pq, (-eff, i, cost))

        print(f"\n{'Seq':<4} {'Gear Name':<30} {'Cost':<5} {'Eff':<5} {'Mission Advice'}")
        print(f"{'-'*95}")

        remaining_budget = self.hammer_budget
        step = 1
        available_denominations = defaultdict(int)

        while pq and remaining_budget > 0:
            neg_eff, idx, cost = heapq.heappop(pq)
            eff = -neg_eff
            piece = self.gear_pieces[idx]

            if cost > remaining_budget:
                continue

            # EXECUTE SIMULATED STEP
            remaining_budget -= cost
            available_denominations[cost] += 1

            # Mission Logic
            advice = ""
            if cost == 10:
                advice = ">> Perfect for [Mission 10]"
            elif cost == 20:
                advice = ">> Perfect for [Mission 20]"
            elif cost == 30:
                advice = ">> Fits [Mission 20] (Waste 10) OR Pair for [Mission 35]"
            elif cost >= 40:
                advice = ">> Perfect for [Mission 35]"

            print(f"{step:<4} {piece.name:<30} {cost:<5} {eff:<5.2f} {advice}")

            # UNLOCK NEXT STEP
            sim_state[idx]['mastery'] += 1
            new_mastery = sim_state[idx]['mastery']

            if self.max_mastery_level is None or new_mastery < self.max_mastery_level:
                new_cost = hammer_cost_for_mastery(new_mastery)
                gain = (sim_state[idx]['base_score'] * 0.10) * sim_state[idx]['multiplier']
                new_eff = gain / new_cost

                if new_eff >= self.min_efficiency:
                    heapq.heappush(pq, (-new_eff, idx, new_cost))

            step += 1

        print(f"{'-'*95}")
        print(f"SUMMARY OF AVAILABLE KEYS (In Efficient Sequence):")
        keys = sorted(available_denominations.keys())
        for k in keys:
            print(f"  {k}-Cost Upgrades: {available_denominations[k]}")
        print(f"{'='*95}\n")


# --- GEAR XP OPTIMIZER (UNCHANGED LOGIC) ---


class GearOptimizer:
    """Optimizes gear upgrade allocation using greedy algorithm."""

    def __init__(self, gear_pieces: List[GearPiece], xp_budget: int):
        self.gear_pieces = gear_pieces
        self.xp_budget = xp_budget
        self.xp_pool = xp_budget
        self.history: List[Dict[str, Any]] = []

    def optimize(self) -> Dict[str, Dict[str, Any]]:
        while self.xp_pool >= 10:
            best_piece = self._find_best_piece()
            if best_piece is None:
                break
            xp_to_apply = best_piece.xp_to_apply()
            leveled_up, _ = best_piece.apply_xp(xp_to_apply)
            self.xp_pool -= xp_to_apply
            if leveled_up:
                self.history.append(
                    {
                        "name": best_piece.name,
                        "new_level": best_piece.level,
                        "xp_applied": xp_to_apply,
                        "weighted_stats": best_piece.weighted_stats_per_level,
                    }
                )
        return self._compile_results()

    def _find_best_piece(self) -> Optional[GearPiece]:
        best_piece = None
        best_efficiency = 0.0
        for piece in self.gear_pieces:
            if not piece.can_upgrade(self.xp_pool):
                continue
            eff = piece.efficiency()
            if eff > best_efficiency:
                best_efficiency = eff
                best_piece = piece
        return best_piece

    def _compile_results(self) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"from": None, "to": 0, "xp": 0, "stats": 0.0, "levels": 0}
        )
        for entry in self.history:
            name = entry["name"]
            if results[name]["from"] is None:
                results[name]["from"] = entry["new_level"] - 1
            results[name]["to"] = entry["new_level"]
            results[name]["xp"] += entry["xp_applied"]
            results[name]["stats"] += entry["weighted_stats"]
            results[name]["levels"] += 1
        return dict(results)

    def print_results(self, results: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        if results is None:
            results = self._compile_results()
        sorted_results = sorted(results.items(), key=lambda x: x[1]["xp"], reverse=True)

        print(f"{'='*95}")
        print(f"GEAR UPGRADE OPTIMIZATION RESULTS")
        print(f"{'='*95}")
        print(f"Total XP Budget: {self.xp_budget:,}")
        print(f"XP Used:       {self.xp_budget - self.xp_pool:,}")
        print(f"XP Remaining:  {self.xp_pool:,}\n")
        print(
            f"{'Gear Piece':<35} {'From→To':<10} {'Levels':<8} "
            f"{'XP Applied':<12} {'Weighted Stats':<15}"
        )
        print(f"{'='*95}")

        # Group by hero for nicer display
        from collections import defaultdict as _dd
        hero_groups: Dict[str, List[Tuple[str, Dict[str, Any]]]] = _dd(list)

        for name, data in sorted_results:
            if data["levels"] > 0:
                hero_name = name.split()[0]
                hero_groups[hero_name].append((name, data))

        sorted_heroes = sorted(
            hero_groups.items(),
            key=lambda x: sum(d["xp"] for _, d in x[1]),
            reverse=True,
        )

        slot_order = {'Helmet': 0, 'Chest': 1, 'Gloves': 2, 'Boots': 3}

        def get_slot_priority(gear_name: str) -> int:
            for slot, priority in slot_order.items():
                if slot in gear_name:
                    return priority
            return 999

        total_stats = 0.0
        for hero, items in sorted_heroes:
            sorted_items = sorted(items, key=lambda x: get_slot_priority(x[0]))
            for name, data in sorted_items:
                print(
                    f"{name:<35} "
                    f"{data['from']:>3}→{data['to']:<3} "
                    f"{data['levels']:<8} "
                    f"{data['xp']:<12,} "
                    f"{data['stats']:<15.1f}"
                )
                total_stats += data["stats"]
            print(f"{'-'*95}")

        print(f"{'='*95}")
        print(
            f"{'TOTAL':<35} {'':<10} {'':<8} "
            f"{self.xp_budget - self.xp_pool:<12,} {total_stats:<15.1f}"
        )


# --- FACTORY FUNCTIONS ---


def create_inventory(stats_db: Dict[str, GearStats]) -> List[GearPiece]:
    """Creates the Inventory with IDs."""

    def make(id: str, name: str, lvl: int, tier: str, stat_key: str, bonus: float = 1.0):
        return GearPiece(id, name, lvl, tier, stats_db[stat_key], bonus)

    return [
        # --- MARLIN'S SET (Gold Archer) ---
        make("marlin_helm", "Marlin Gold Arc Helmet", 76, 'Gold', 'Gold_Arc_HB', 1.50),
        make("marlin_chest", "Marlin Gold Arc Chest", 55, 'Gold', 'Gold_Arc_CG', 1.20),
        make("marlin_glove", "Marlin Gold Arc Gloves", 55, 'Gold', 'Gold_Arc_CG', 1.20),
        make("marlin_boots", "Marlin Gold Arc Boots", 74, 'Gold', 'Gold_Arc_HB', 1.50),

        # --- JABEL'S SET (Gold Cav) ---
        make("jabel_helm", "Jabel Gold Cav Helmet", 50, 'Gold', 'Gold_Cav_HB', 1.20),
        make("jabel_chest", "Jabel Gold Cav Chest", 45, 'Gold', 'Gold_Cav_CG', 1.0),
        make("jabel_glove", "Jabel Gold Cav Gloves", 45, 'Gold', 'Gold_Cav_CG', 1.0),
        make("jabel_boots", "Jabel Gold Cav Boots", 50, 'Gold', 'Gold_Cav_HB', 1.20),

        # --- HOWARD'S SET (Mix Inf) ---
        make("howard_helm", "Howard Purple Inf Helmet", 31, 'Purple', 'Purple_Inf_HB', 1.0),
        make("howard_chest", "Howard Gold Inf Chest", 49, 'Gold', 'Gold_Inf_CG', 1.0),
        make("howard_glove", "Howard Gold Inf Gloves", 49, 'Gold', 'Gold_Inf_CG', 1.0),
        make("howard_boots", "Howard Purple Inf Boots", 30, 'Purple', 'Purple_Inf_HB', 1.0),

        # --- ZOE'S SET (Gold Inf) ---
        make("zoe_helm", "Zoe Gold Inf Helmet", 56, 'Gold', 'Gold_Inf_HB', 1.0),
        make("zoe_chest", "Zoe Gold Inf Chest", 66, 'Gold', 'Gold_Inf_CG', 1.30),
        make("zoe_glove", "Zoe Gold Inf Gloves", 66, 'Gold', 'Gold_Inf_CG', 1.30),
        make("zoe_boots", "Zoe Gold Inf Boots", 56, 'Gold', 'Gold_Inf_HB', 1.0),

        # --- DIANA'S SET (Gold Archer) ---
        make("diana_helm", "Diana Gold Arc Helmet", 40, 'Gold', 'Gold_Arc_HB', 1.0),
        make("diana_chest", "Diana Gold Arc Chest", 41, 'Gold', 'Gold_Arc_CG', 1.0),
        make("diana_glove", "Diana Gold Arc Gloves", 41, 'Gold', 'Gold_Arc_CG', 1.0),
        make("diana_boots", "Diana Gold Arc Boots", 40, 'Gold', 'Gold_Arc_HB', 1.0),
    ]


# --- MAIN SCRIPT ---


def main():
    # =============================================================================
    # A) DEFINE OBJECTS
    # =============================================================================

    # 1. SETUP STATS & INVENTORY
    #
    # Base (pre-Forgehammer) stats per level for each gear type.
    # Format:
    #   GearStats(Hero Atk, Hero Def, Hero HP,
    #             Escort Atk, Escort Def, Escort HP,
    #             Exped Lethality, Exped Health)
    #
    # In-game, some stats belong to the same "bucket" (e.g., Hero Def + Hero HP)
    # and their weights should sum to a meaningful category weight.
    # CG (Chest/Gloves) = primarily Def/HP, HB (Helmet/Boots) = Atk.
    # Expedition stats (Lethality/Health) are used in Expedition scenarios.
    GEAR_STATS: Dict[str, GearStats] = {
        # --- INFANTRY ---
        # Gold
        'Gold_Inf_CG': GearStats(0.0, 2.30, 22.50, 0.0, 0.77, 7.50, 0.0, 0.35),
        'Gold_Inf_HB': GearStats(3.00, 0.0, 22.50, 1.00, 0.0, 7.50, 0.35, 0.0),
        # Purple
        'Purple_Inf_CG': GearStats(0.0, 1.35, 13.50, 0.0, 0.50, 4.50, 0.0, 0.21),
        'Purple_Inf_HB': GearStats(1.80, 0.0, 9.00, 0.60, 0.0, 3.00, 0.21, 0.0),
        # Blue
        'Blue_Inf_CG': GearStats(0.0, 1.20, 6.00, 0.0, 0.40, 2.00, 0.0, 0.14),
        'Blue_Inf_HB': GearStats(1.20, 0.0, 9.00, 0.40, 0.0, 3.00, 0.14, 0.0),

        # --- CAVALRY ---
        # Gold
        'Gold_Cav_CG': GearStats(0.0, 3.64, 11.25, 0.0, 1.22, 3.75, 0.0, 0.35),
        'Gold_Cav_HB': GearStats(3.00, 0.0, 11.25, 1.00, 0.0, 3.75, 0.35, 0.0),
        # Purple
        'Purple_Cav_CG': GearStats(0.0, 2.20, 6.75, 0.0, 0.75, 2.25, 0.0, 0.21),
        'Purple_Cav_HB': GearStats(1.80, 0.0, 6.75, 0.60, 0.0, 2.25, 0.26, 0.0),
        # Blue
        'Blue_Cav_CG': GearStats(0.0, 1.20, 4.50, 0.0, 0.40, 1.50, 0.0, 0.14),
        'Blue_Cav_HB': GearStats(1.20, 0.0, 4.50, 0.40, 0.0, 1.50, 0.14, 0.0),

        # --- ARCHER ---
        # Gold
        'Gold_Arc_CG': GearStats(0.0, 3.00, 15.00, 0.0, 1.00, 5.00, 0.0, 0.35),
        'Gold_Arc_HB': GearStats(3.00, 0.0, 15.00, 1.00, 0.0, 5.00, 0.35, 0.0),
        # Purple
        'Purple_Arc_CG': GearStats(0.0, 1.80, 9.00, 0.0, 0.60, 3.00, 0.0, 0.21),
        'Purple_Arc_HB': GearStats(1.80, 0.0, 9.00, 0.60, 0.0, 3.00, 0.21, 0.0),
        # Blue
        'Blue_Arc_CG': GearStats(0.0, 1.20, 6.00, 0.0, 0.40, 2.00, 0.0, 0.14),
        'Blue_Arc_HB': GearStats(1.20, 0.0, 6.00, 0.40, 0.0, 2.00, 0.14, 0.0),
    }

    inventory = create_inventory(GEAR_STATS)
    inventory_registry = {p.id: p for p in inventory}

    # 2. SETUP BASELINE HERO STAT PROFILES
    #
    # Format / ordering:
    #   (Hero Atk, Hero Def, Hero HP,
    #    Escort Atk, Escort Def, Escort HP,
    #    Exped Lethality, Exped Health)
    #
    # Important:
    #   - In-game, Hero Def + Hero HP share a bucket; same for Escort Def + Escort HP, etc.
    #   - The sum of weights for a bucket reflects that bucket's overall importance.
    #   - These are baseline per-use-case; scenarios may override some stats locally.

    raw_profiles = {
        'Marlin': (8.0, 0.7, 0.7, 2.0, 0.7, 0.7, 5.0, 2.0),
        'Diana': (1.0, 1.0, 1.1, 1.0, 1.1, 1.0, 0.0, 0.0),
        'Jabel': (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.5),
        'Zoe': (0.1, 1.0, 1.0, 0.05, 1.0, 1.0, 2.0, 6.0),
        'Howard': (0.0, 0.85, 0.85, 0.0, 0.6, 0.6, 0.0, 0.0),
        'Chenko': (0.8, 0.8, 0.55, 0.75, 0.75, 0.75, 0.0, 0.0),
        'Seth': (0.0, 0.0, 0.4, 0.05, 0.05, 0.85, 0.0, 0.0),
    }

    baseline_profiles = {name: StatProfile(data) for name, data in raw_profiles.items()}

    # Optional per-scenario overrides:
    # Dict[scenario_index, Dict[HeroName, Dict[stat_name, float]]]
    # Example (commented):
    #
    # scenario_profile_overrides = {
    #     1: {  # scenario index 1
    #         'Marlin': {'exped_health': 4.0},
    #         'Zoe': {'exped_lethality': 3.0},
    #     }
    # }
    scenario_profile_overrides: Dict[int, Dict[str, Dict[str, float]]] = {}

    # 3. SETUP LOADOUTS
    manager = LoadoutManager()

    # A. Expedition (Base)
    exp_loadout = manager.create("Expedition")

    # Generic Assignment: Everyone wears their own gear
    active_heroes = ['Marlin', 'Diana', 'Jabel', 'Howard', 'Zoe']
    for hero in active_heroes:
        prefix = hero.lower()
        exp_loadout.assign(hero, "Helmet", f"{prefix}_helm")
        exp_loadout.assign(hero, "Chest", f"{prefix}_chest")
        exp_loadout.assign(hero, "Gloves", f"{prefix}_glove")
        exp_loadout.assign(hero, "Boots", f"{prefix}_boots")

    # B. Conquest (Derived from Expedition)
    conq_loadout = manager.fork("Expedition", "Conquest")

    # Special Logic: "Put Archer Def gears on Diana"
    # We swap ONLY Chest/Gloves. Marlin keeps his Helmet/Boots.
    manager.swap_slots("Conquest", "Marlin", "Diana", ["Chest", "Gloves"])

    # Integrity checks (optional, but helpful)
    for lo in [exp_loadout, conq_loadout]:
        ok, errors = lo.validate_integrity()
        if not ok:
            print(f"[WARN] Loadout {lo.name} has integrity issues:")
            for e in errors:
                print("  -", e)

    # 4. SCENARIO CONFIGURATION
    #
    # Each scenario is defined by:
    #   - A loadout (who wears what).
    #   - A GameMode (which stats are active).
    #   - An optional scenario_weight (importance among scenarios).
    #
    # Engine enforces:
    #   - Either ALL scenarios have explicit weights, or NONE do (defaults to equal).
    active_scenarios: List[Tuple[Loadout, set]] = [
        (conq_loadout, GameMode.CONQUEST),
        (exp_loadout, GameMode.EXPEDITION),
    ]

    # Optional, incremental scenario weights:
    #   - Initialize to None to allow setting them one by one.
    #   - Before execution, either all must be set, or all left as None.
    scenario_weights: List[Optional[float]] = [None] * len(active_scenarios)
    # Example (uncomment to enable explicit weights):
    # scenario_weights[0] = 0.7
    # scenario_weights[1] = 0.3

    # 5. FORGEHAMMER STRATEGY CONFIGURATION
    #
    # FORGE_STRATEGY:
    #   Dict[HeroName] -> (Helmet, Chest, Gloves, Boots)
    #
    # Multipliers (per slot, per hero, mapped through the canonical loadout):
    #   0.0  = BANNED        -> this gear never receives hammers
    #   0.5  = Low priority  -> allowed, but only if higher-multiplier options are exhausted
    #   1.0  = Normal        -> standard investment level
    #   2.0+ = High priority -> very strong long‑term candidate for hammers
    #
    # Relationship to XP/scenario scoring:
    #   - Gear *value* is already determined by:
    #       • gear stat lines (GearStats),
    #       • which hero wears the gear in each scenario (loadouts),
    #       • that hero’s stat weights in each scenario (profiles + overrides),
    #       • and scenario weights.
    #   - Forgehammer multipliers are an additional, *separate* layer that control
    #     which Gold pieces are even eligible for mastery, and how aggressively
    #     they should be pushed, given that mastery forges are irreversible and forgehammers scarce.
    #
    # Mapping from this table to actual gears:
    #   - A single, explicit canonical loadout is chosen for forgehammer.
    #   - For each hero in FORGE_STRATEGY, the (Helm, Chest, Gloves, Boots) tuple
    #     is applied to whatever specific gear IDs that hero is wearing in the
    #     canonical loadout.
    #   - The resulting per‑gear hammer_multiplier is what the hammer optimizer
    #     actually uses; other scenarios/loadouts do NOT affect hammer eligibility.
    #
    FORGE_STRATEGY: Dict[str, Tuple[float, float, float, float]] = {
        # Marlin’s offensive pieces (Helm/Boots) are long‑term forge targets;
        # his defensive pieces (Chest/Gloves) are still allowed, but at normal weight.
        'Marlin': (2.0, 1.0, 1.0, 2.0),

        # Zoe’s Atk pieces (Helm/Boots) are niche: only useful for rally leader in Expedition
        # and HP stat‑sticks in Conquest, both comparatively niche, so they get
        # a reduced multiplier. Her defensive pieces are fully acceptable forges.
        'Zoe': (0.5, 1.0, 1.0, 0.5),

        # Jabel’s offensive pieces are allowed; his defensive pieces are
        # explicitly banned from hammer investment.
        'Jabel': (1.0, 0.0, 0.0, 1.0),

        # Unlisted heroes (e.g. Diana) default to 0.0 (BANNED).
    }

    # Choose which loadout is canonical for Forgehammer.
    # Only this loadout controls hammer eligibility. Other scenarios/loadouts
    # affect XP/composite scores but *not* hammer config.
    CANONICAL_FORGE_LOADOUT = exp_loadout

    # --- SIMULATION CONFIG ---
    hammer_budget = 59
    max_mastery_level: Optional[int] = None
    HAMMER_THRESHOLD_MODE = "RELATIVE"  # or "ABSOLUTE"
    HAMMER_THRESHOLD_VALUE = 0.8        # if RELATIVE: fraction of peak efficiency

    # XP Phase budget
    xp_budget = 24360

    # =============================================================================
    # B) EXECUTION
    # =============================================================================

    # 1. Build scenarios with weights (enforce all-or-none at execution)
    if any(w is not None for w in scenario_weights):
        # Enforce completeness
        assert all(
            w is not None for w in scenario_weights
        ), "Either specify ALL scenario weights or leave ALL as None."
        scenarios: List[Tuple[Loadout, set, float]] = [
            (lo, mode, float(w))
            for (lo, mode), w in zip(active_scenarios, scenario_weights)
        ]
    else:
        # No explicit weights: equal weight (1.0) will be injected by engine
        scenarios = active_scenarios  # type: ignore[assignment]

    # 2. CONFIG ECHO
    print("\n" + "=" * 95)
    print("CONFIG ECHO")
    print("=" * 95)
    print("Scenarios:")
    tmp_scenarios: List[Tuple[Loadout, set, float]]
    # Normalize for echo
    if isinstance(scenarios[0], tuple) and len(scenarios[0]) == 3:  # type: ignore[index]
        tmp_scenarios = [(s[0], s[1], float(s[2])) for s in scenarios]  # type: ignore[list-item]
    else:
        tmp_scenarios = [(s[0], s[1], 1.0) for s in scenarios]  # type: ignore[list-item]
    for s in tmp_scenarios:
        lo, mode, w = s
        print(f" - {lo.name:<12} | mode_stats={len(mode):<2} | weight={w}")
    print(f"Canonical forge loadout: {CANONICAL_FORGE_LOADOUT.name}")
    print(f"Hammer budget: {hammer_budget} | threshold: {HAMMER_THRESHOLD_MODE} {HAMMER_THRESHOLD_VALUE}")
    print(f"XP budget: {xp_budget}")
    print("=" * 95 + "\n")

    # 3. SCORING ENGINE: Compute composite_weight_score per gear
    engine = GameScoringEngine(
        inventory_registry,
        baseline_profiles,
        scenario_profile_overrides=scenario_profile_overrides,
    )
    engine.calculate_utilities(scenarios)  # type: ignore[arg-type]

    # 4. APPLY FORGEHAMMER STRATEGY
    apply_forge_strategy(inventory_registry, CANONICAL_FORGE_LOADOUT, FORGE_STRATEGY)

    # 5. HAMMER OPTIMIZATION
    hammer_opt = ForgehammerOptimizer(
        inventory,
        hammer_budget,
        max_mastery_level=max_mastery_level,
    )

    # Peak efficiency calculation respects hammer_multiplier (0.0 banned, Gold-only)
    peak_efficiency = hammer_opt.calculate_peak_efficiency()
    if HAMMER_THRESHOLD_MODE == "RELATIVE":
        min_eff = peak_efficiency * HAMMER_THRESHOLD_VALUE
    else:
        min_eff = HAMMER_THRESHOLD_VALUE
    hammer_opt.min_efficiency = min_eff

    print(f"Hammer Optimization Init: Peak Eff={peak_efficiency:.4f}, Threshold={min_eff:.4f}")
    hammer_opt.generate_strategic_roadmap()

    # 6. XP OPTIMIZATION
    optimizer = GearOptimizer(inventory, xp_budget)
    results = optimizer.optimize()
    optimizer.print_results(results)


if __name__ == "__main__":
    main()
