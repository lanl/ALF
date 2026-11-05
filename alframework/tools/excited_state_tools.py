"""Small, dependency-free contracts shared by excited-state components."""

from __future__ import annotations

import math
import re
from typing import Any


_ENERGY_KEY = re.compile(r"^sE(\d+)$")
_FORCE_KEY = re.compile(r"^F(\d+)$")
_GAP_KEY = re.compile(r"^dE(?:(\d)(\d)|(\d+)_(\d+))$")


def derive_state_property_table(
    properties_list: dict[str, list[Any]],
    *,
    require_forces: bool = False,
) -> list[dict[str, Any]]:
    """Validate flattened ``sE#``/``F#`` properties and describe each state."""

    if not isinstance(properties_list, dict) or not properties_list:
        raise ValueError("properties_list must be a non-empty dictionary.")

    energies: dict[int, dict[str, Any]] = {}
    forces: dict[int, dict[str, Any]] = {}
    for key, schema in properties_list.items():
        if not isinstance(schema, (list, tuple)) or not schema:
            raise ValueError(f"Property schema for {key!r} must be a non-empty list.")
        energy_match = _ENERGY_KEY.fullmatch(str(key))
        force_match = _FORCE_KEY.fullmatch(str(key))
        if energy_match:
            state = int(energy_match.group(1))
            energies[state] = {
                "state": state,
                "energy_key": str(key),
                "energy_db_name": str(schema[0]),
            }
        elif force_match:
            state = int(force_match.group(1))
            forces[state] = {
                "force_key": str(key),
                "force_db_name": str(schema[0]),
            }

    if not energies:
        raise ValueError(
            "properties_list must define excited-state energies named sE0, sE1, ..."
        )
    states = sorted(energies)
    if states != list(range(states[-1] + 1)):
        raise ValueError(
            f"Excited-state energy keys must be contiguous from zero; found {states}."
        )

    rows: list[dict[str, Any]] = []
    for state in states:
        row = dict(energies[state])
        if state in forces:
            row.update(forces[state])
        elif require_forces:
            raise ValueError(f"Missing excited-state force property F{state}.")
        else:
            row.update({"force_key": None, "force_db_name": None})
        rows.append(row)
    return rows


def gap_key_for_pair(lower_state: int, upper_state: int) -> str:
    """Return ALF's flattened property key for one directed state gap."""

    lower = int(lower_state)
    upper = int(upper_state)
    if 0 <= lower <= 9 and 0 <= upper <= 9:
        return f"dE{lower}{upper}"
    return f"dE{lower}_{upper}"


def parse_gap_key(property_key: str) -> tuple[int, int] | None:
    """Parse ``dE01`` or the unambiguous multi-digit form ``dE0_10``."""

    match = _GAP_KEY.fullmatch(str(property_key))
    if match is None:
        return None
    if match.group(1) is not None:
        lower, upper = int(match.group(1)), int(match.group(2))
    else:
        lower, upper = int(match.group(3)), int(match.group(4))
    if lower == upper:
        raise ValueError(
            f"Gap property {property_key!r} cannot reference one state twice."
        )
    return lower, upper


def _property_scale(
    properties_list: dict[str, list[Any]], property_key: str
) -> float:
    schema = properties_list[property_key]
    scale = float(schema[2]) if len(schema) > 2 else 1.0
    if not math.isfinite(scale):
        raise ValueError(
            f"Property scale for {property_key!r} must be finite."
        )
    return scale


def _gap_state_index(value: Any, *, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be a nonnegative integer.")
    try:
        state = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{context} must be a nonnegative integer; received {value!r}."
        ) from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(
            f"{context} must be a nonnegative integer; received {value!r}."
        )
    return state


def derive_gap_property_table(
    properties_list: dict[str, list[Any]],
    *,
    gap_config: dict[str, Any] | None = None,
    require_properties: bool = False,
) -> list[dict[str, Any]]:
    """Validate configured ``dE#`` targets and describe their state pairs.

    Explicit ``gap_config.pairs`` selects a subset. When pairs are omitted,
    every gap property present in ``properties_list`` is returned.
    """

    state_table = derive_state_property_table(properties_list)
    available_states = {int(row["state"]) for row in state_table}
    config = dict(gap_config or {})
    configured_pairs = config.get("pairs")
    if configured_pairs is None:
        pair_items: list[tuple[int, int]] = []
        for property_key in properties_list:
            pair = parse_gap_key(str(property_key))
            if pair is not None:
                pair_items.append(pair)
    else:
        if not isinstance(configured_pairs, (list, tuple)):
            raise TypeError("gap target pairs must be a list of state pairs.")
        pair_items = []
        for index, pair in enumerate(configured_pairs):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(
                    "Each gap target pair must contain exactly two states; "
                    f"entry {index} is {pair!r}."
                )
            pair_items.append(
                (
                    _gap_state_index(
                        pair[0], context=f"gap target pair {index}[0]"
                    ),
                    _gap_state_index(
                        pair[1], context=f"gap target pair {index}[1]"
                    ),
                )
            )

    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for lower, upper in pair_items:
        lower, upper = int(lower), int(upper)
        if lower < 0 or upper < 0:
            raise ValueError(
                f"Gap target states must be nonnegative; received {(lower, upper)}."
            )
        if lower >= upper:
            raise ValueError(
                "Gap targets must be ordered from lower to higher state; "
                f"received {(lower, upper)}."
            )
        pair = (lower, upper)
        if pair in seen:
            raise ValueError(f"Duplicate gap target pair {pair}.")
        seen.add(pair)
        if lower not in available_states or upper not in available_states:
            raise ValueError(
                f"Gap target {pair} references states outside "
                f"{sorted(available_states)}."
            )

        gap_key = gap_key_for_pair(lower, upper)
        if gap_key not in properties_list:
            if require_properties:
                raise ValueError(
                    f"Gap target {pair} requires {gap_key!r} in properties_list."
                )
            continue
        gap_schema = properties_list[gap_key]
        if not isinstance(gap_schema, (list, tuple)) or len(gap_schema) < 2:
            raise ValueError(
                f"Gap property schema for {gap_key!r} must include database "
                "name and property kind."
            )
        if str(gap_schema[1]).strip().lower() != "system":
            raise ValueError(
                f"Gap property {gap_key!r} must be a system property."
            )
        gap_db_name = str(gap_schema[0])
        if not gap_db_name:
            raise ValueError(
                f"Gap property {gap_key!r} requires a nonempty database name."
            )

        lower_key = f"sE{lower}"
        upper_key = f"sE{upper}"
        lower_scale = _property_scale(properties_list, lower_key)
        upper_scale = _property_scale(properties_list, upper_key)
        gap_scale = _property_scale(properties_list, gap_key)
        if not (
            math.isclose(lower_scale, upper_scale, rel_tol=0.0, abs_tol=0.0)
            and math.isclose(lower_scale, gap_scale, rel_tol=0.0, abs_tol=0.0)
        ):
            raise ValueError(
                f"Gap property {gap_key!r} and state energies {lower_key!r}/"
                f"{upper_key!r} must use identical property scales."
            )
        rows.append(
            {
                "lower_state": lower,
                "upper_state": upper,
                "lower_energy_key": lower_key,
                "upper_energy_key": upper_key,
                "gap_key": gap_key,
                "gap_db_name": gap_db_name,
                "scale": gap_scale,
            }
        )
    return rows
