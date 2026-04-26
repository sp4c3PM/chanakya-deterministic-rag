"""
Agentic tools — callable by the LLM via tool use.
These turn retrieved Atom data into real-world decisions.
"""
import math

# Approximate coordinates for key Indian strategic locations
LOCATIONS = {
    "delhi": (28.6, 77.2),
    "mumbai": (19.1, 72.9),
    "chennai": (13.1, 80.3),
    "jaisalmer": (26.9, 70.9),
    "leh": (34.2, 77.6),
    "port blair": (11.7, 92.7),
    "karachi": (24.9, 67.0),
    "lahore": (31.5, 74.3),
    "islamabad": (33.7, 73.1),
    "beijing": (39.9, 116.4),
    "colombo": (6.9, 79.9),
    "dhaka": (23.8, 90.4),
    # Additional strategic targets
    "lhasa": (29.6, 91.1),
    "chengdu": (30.6, 104.1),
    "doklam": (27.2, 89.1),
    "gilgit": (35.9, 74.3),
    "skardu": (35.3, 75.5),
    "rawalpindi": (33.6, 73.1),
}

# --- Unit Normalization ---
# Static exchange rates — documented assumption, version-controlled, air-gapped safe.
# Update this table when rates shift materially (>5%). Last updated: 2026-04-26.
UNIT_RATES_TO_CRORE = {
    "inr_crore": 1.0,
    "inr_million": 0.1,        # 10M INR = 1 Cr → 1M INR = 0.1 Cr
    "inr_lakh": 0.01,          # 100L = 1 Cr → 1L = 0.01 Cr
    "usd_million": 8.5,        # 1M USD × Rs85/USD ÷ 10M(per Cr) = 8.5 Cr
    "usd_billion": 8500.0,     # 1B USD = 1000M USD = 8500 Cr
    "eur_million": 9.2,        # 1M EUR × Rs92/EUR ÷ 10M = 9.2 Cr
    "gbp_million": 10.7,       # 1M GBP × Rs107/GBP ÷ 10M = 10.7 Cr
}

def normalize_to_crore(value: float, unit: str) -> float:
    """Convert any financial value to INR Crore. Raises ValueError for unknown units."""
    rate = UNIT_RATES_TO_CRORE.get(unit.lower().replace(" ", "_"))
    if rate is None:
        raise ValueError(
            f"Unknown unit '{unit}'. Known: {list(UNIT_RATES_TO_CRORE.keys())}"
        )
    return round(value * rate, 2)


# --- Physics ---
def speed_of_sound_ms(altitude_m: float = 0.0) -> float:
    """
    ISA standard atmosphere speed of sound.
    Troposphere only (valid to 11,000m). Above 11km, temperature is constant at -56.5°C.
    Sea level (0m): ~340 m/s. Cruise altitude (10,000m): ~299 m/s.
    """
    T_sea = 288.15        # K at sea level (ISA, 15°C)
    lapse = 6.5e-3        # K/m (ISA tropospheric lapse rate)
    alt_clamped = min(altitude_m, 11000.0)
    T = T_sea - lapse * alt_clamped
    return round(331.3 * math.sqrt(T / 273.15), 2)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def range_check(platform_range_km: float, origin: str, target: str) -> dict:
    """Check if a platform can reach a target from an origin given its maximum range."""
    o = LOCATIONS.get(origin.lower())
    t = LOCATIONS.get(target.lower())

    if not o:
        return {"error": f"Unknown origin: {origin}. Known: {list(LOCATIONS.keys())}"}
    if not t:
        return {"error": f"Unknown target: {target}. Known: {list(LOCATIONS.keys())}"}

    dist = round(haversine_km(*o, *t), 1)
    feasible = dist <= platform_range_km
    margin = round(platform_range_km - dist, 1)

    return {
        "origin": origin,
        "target": target,
        "distance_km": dist,
        "platform_range_km": platform_range_km,
        "feasible": feasible,
        "margin_km": margin,
        "verdict": f"{'IN RANGE' if feasible else 'OUT OF RANGE'} — distance {dist} km, range {platform_range_km} km, margin {margin:+} km"
    }


def ic_compliance_check(
    ic_percent: float,
    procurement_category: str,
    required_ic: float = None,
    policy_source: str = "DAP 2020 (hardcoded)"
) -> dict:
    """
    Check if a platform's indigenous content meets the requirement for a procurement category.

    required_ic: if provided by the agent (RAG-retrieved from corpus), overrides the hardcoded table.
                 Pass this when the corpus contains a more current policy version (e.g. DAP 2026 draft).
    policy_source: document the source of the threshold so the audit trace is transparent.
    """
    # Hardcoded DAP 2020 thresholds — fallback only.
    # If corpus has a different (newer) value, pass required_ic explicitly.
    DAP2020_REQUIREMENTS = {
        "buy indian-iddm": 50.0,
        "buy indian iddm": 50.0,
        "buy (indian-iddm)": 50.0,
        "buy indian": 40.0,
        "buy (indian)": 40.0,
        "buy and make indian": 50.0,
        "buy and make (indian)": 50.0,
        "buy global": 0.0,
    }

    if required_ic is not None:
        # RAG-retrieved threshold takes precedence
        required = required_ic
    else:
        cat_lower = procurement_category.lower()
        required = DAP2020_REQUIREMENTS.get(cat_lower)
        if required is None:
            return {"error": f"Unknown category: {procurement_category}. Known: {list(DAP2020_REQUIREMENTS.keys())}"}
        policy_source = "DAP 2020 (hardcoded)"

    compliant = ic_percent >= required
    return {
        "category": procurement_category,
        "required_ic_percent": required,
        "actual_ic_percent": ic_percent,
        "compliant": compliant,
        "policy_source": policy_source,
        "verdict": f"{'COMPLIANT' if compliant else 'NON-COMPLIANT'} — {ic_percent}% vs required {required}% [{policy_source}]"
    }


def budget_check(
    quantity: int,
    budget_crore: float,
    unit_cost_crore: float = None,
    unit_cost_raw: float = None,
    unit_cost_unit: str = None,
) -> dict:
    """
    Check if procuring a quantity of a platform fits within a budget.

    unit_cost_crore: pre-normalized cost in INR Crore. Use this directly if the doc quotes crore.
    unit_cost_raw + unit_cost_unit: alternative — pass raw value and unit string; tool normalizes.
    budget_crore: available budget, always in INR Crore.
    """
    # Normalize if raw value + unit provided
    if unit_cost_raw is not None and unit_cost_unit is not None:
        try:
            unit_cost_crore = normalize_to_crore(unit_cost_raw, unit_cost_unit)
        except ValueError as e:
            return {"error": str(e)}
    elif unit_cost_crore is None:
        return {"error": "Provide either unit_cost_crore or both unit_cost_raw + unit_cost_unit"}

    total = round(unit_cost_crore * quantity, 2)
    feasible = total <= budget_crore
    shortfall = round(total - budget_crore, 2)

    return {
        "unit_cost_crore": unit_cost_crore,
        "quantity": quantity,
        "total_cost_crore": total,
        "budget_crore": budget_crore,
        "feasible": feasible,
        "shortfall_crore": shortfall if not feasible else 0,
        "verdict": f"{'FEASIBLE' if feasible else 'OVER BUDGET'} — total Rs {total} Cr vs budget Rs {budget_crore} Cr"
    }


def calculate_impact_time(
    distance_km: float,
    mach_speed: float,
    altitude_m: float = 10000.0,
) -> dict:
    """
    Calculate time to impact for a missile given distance, speed (Mach), and cruise altitude.

    altitude_m: cruise altitude in metres. Defaults to 10,000m (BrahMos cruise profile).
                Speed of sound varies from ~340 m/s (sea level) to ~299 m/s (10km).
                Using sea level (343 m/s) would underestimate impact time by ~13% at cruise altitude.
    """
    c = speed_of_sound_ms(altitude_m)
    speed_ms = mach_speed * c
    speed_kmh = round(speed_ms * 3.6, 1)
    time_seconds = round((distance_km * 1000) / speed_ms, 1)
    time_minutes = round(time_seconds / 60, 2)

    return {
        "distance_km": distance_km,
        "mach_speed": mach_speed,
        "altitude_m": altitude_m,
        "speed_of_sound_ms": c,
        "speed_kmh": speed_kmh,
        "time_seconds": time_seconds,
        "time_minutes": time_minutes,
        "verdict": f"At Mach {mach_speed} / {altitude_m/1000:.0f}km alt, impact at {distance_km}km in {time_seconds}s ({time_minutes} min)"
    }


# Tool registry — all 4 tools defined before this list is constructed
TOOLS = [
    {
        "name": "range_check",
        "description": "Check if a platform can reach a target from an origin given its maximum range",
        "parameters": {
            "platform_range_km": "Maximum range of the platform in km (float)",
            "origin": "Launch location name (e.g. 'jaisalmer', 'delhi')",
            "target": "Target location name (e.g. 'karachi', 'beijing')"
        },
        "fn": range_check
    },
    {
        "name": "ic_compliance_check",
        "description": (
            "Check if a platform's indigenous content % meets the procurement category requirement. "
            "Pass required_ic and policy_source if the corpus contains a more current threshold than DAP 2020."
        ),
        "parameters": {
            "ic_percent": "Indigenous content percentage (float)",
            "procurement_category": "DAP category (e.g. 'Buy Indian-IDDM')",
            "required_ic": "RAG-retrieved IC% threshold (float, optional — overrides hardcoded table)",
            "policy_source": "Source document for the threshold (string, optional)"
        },
        "fn": ic_compliance_check
    },
    {
        "name": "budget_check",
        "description": (
            "Check if procuring a quantity of a platform fits within a budget. "
            "Pass unit_cost_raw + unit_cost_unit when the doc quotes non-INR units (USD, EUR, INR Million)."
        ),
        "parameters": {
            "quantity": "Number of units (int)",
            "budget_crore": "Available budget in INR Crore (float)",
            "unit_cost_crore": "Cost per unit in INR Crore (float) — use if doc quotes crore directly",
            "unit_cost_raw": "Raw cost value if unit is not crore (float, optional)",
            "unit_cost_unit": "Unit string: inr_million / usd_million / eur_million / gbp_million (optional)"
        },
        "fn": budget_check
    },
    {
        "name": "calculate_impact_time",
        "description": (
            "Calculate time for a missile to reach a target given distance, Mach speed, and cruise altitude. "
            "Default altitude is 10,000m (BrahMos cruise profile)."
        ),
        "parameters": {
            "distance_km": "Distance to target in km (float)",
            "mach_speed": "Speed in Mach number (float, e.g. 2.8)",
            "altitude_m": "Cruise altitude in metres (float, optional — default 10000)"
        },
        "fn": calculate_impact_time
    },
]
