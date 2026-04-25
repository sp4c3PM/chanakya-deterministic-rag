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
}

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def range_check(platform_range_km: float, origin: str, target: str) -> dict:
    """
    Check if a platform with a given range can reach a target from an origin.
    Returns distance, feasibility, and margin.
    """
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

def ic_compliance_check(ic_percent: float, procurement_category: str) -> dict:
    """
    Check if a platform's indigenous content meets the DAP 2020 requirement
    for a given procurement category.
    """
    requirements = {
        "buy indian-iddm": 50.0,
        "buy indian iddm": 50.0,
        "buy (indian-iddm)": 50.0,
        "buy indian": 40.0,
        "buy (indian)": 40.0,
        "buy and make indian": 50.0,
        "buy and make (indian)": 50.0,
        "buy global": 0.0,
    }
    cat_lower = procurement_category.lower()
    required = requirements.get(cat_lower)

    if required is None:
        return {"error": f"Unknown category: {procurement_category}. Known: {list(requirements.keys())}"}

    compliant = ic_percent >= required
    return {
        "category": procurement_category,
        "required_ic_percent": required,
        "actual_ic_percent": ic_percent,
        "compliant": compliant,
        "verdict": f"{'COMPLIANT' if compliant else 'NON-COMPLIANT'} — {ic_percent}% vs required {required}%"
    }

def budget_check(unit_cost_crore: float, quantity: int, budget_crore: float) -> dict:
    """
    Check if procuring a given quantity of a platform fits within a budget.
    """
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


# Tool registry for LLM tool-use
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
        "description": "Check if a platform's indigenous content % meets DAP 2020 requirements for a procurement category",
        "parameters": {
            "ic_percent": "Indigenous content percentage (float)",
            "procurement_category": "DAP 2020 category (e.g. 'Buy Indian-IDDM')"
        },
        "fn": ic_compliance_check
    },
    {
        "name": "budget_check",
        "description": "Check if procuring a quantity of a platform fits within a budget",
        "parameters": {
            "unit_cost_crore": "Cost per unit in crore rupees (float)",
            "quantity": "Number of units (int)",
            "budget_crore": "Available budget in crore rupees (float)"
        },
        "fn": budget_check
    }
]

def calculate_impact_time(distance_km: float, mach_speed: float) -> dict:
    """
    Calculate time to impact for a missile given distance and speed in Mach.
    Speed of sound at sea level = 343 m/s.
    """
    speed_ms = mach_speed * 343
    speed_kmh = speed_ms * 3.6
    time_seconds = round((distance_km * 1000) / speed_ms, 1)
    time_minutes = round(time_seconds / 60, 2)

    return {
        "distance_km": distance_km,
        "mach_speed": mach_speed,
        "speed_kmh": round(speed_kmh, 1),
        "time_seconds": time_seconds,
        "time_minutes": time_minutes,
        "verdict": f"At Mach {mach_speed}, impact at {distance_km}km in {time_seconds}s ({time_minutes} min)"
    }

# Add to registry
TOOLS.append({
    "name": "calculate_impact_time",
    "description": "Calculate time for a missile to reach a target given distance and Mach speed",
    "parameters": {
        "distance_km": "Distance to target in km (float)",
        "mach_speed": "Speed in Mach number (float, e.g. 2.8)"
    },
    "fn": calculate_impact_time
})
