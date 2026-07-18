"""
Model registry: maps name → (EpiModel class, description, outputs).

Usage
-----
    from episurveil.models.registry import get_model, list_models, MODEL_INFO
    model = get_model("SEIR")(N=1e6, ...)
    list_models()                  # prints the table
"""
from __future__ import annotations

from episurveil.models.sir     import SIRModel
from episurveil.models.seir    import SEIRModel
from episurveil.models.seird   import SEIRDModel
from episurveil.models.seirv   import SEIRVModel
from episurveil.models.seiarv  import SEIARVModel
from episurveil.models.seirhd  import SEIRHDModel

# SVEAIHCRD is imported lazily to avoid circular dependency with existing code
def _sveaihcrd():
    from episurveil.models.sveaihcrd_model import SVEAIHCRDModel
    return SVEAIHCRDModel


MODEL_INFO: dict[str, dict] = {
    "SIR": {
        "class":      SIRModel,
        "states":     ["S", "I", "R"],
        "log_rw":     ["beta_t"],
        "exogenous":  [],
        "obs_channels": ["cases"],
        "outputs": [
            "S_t, I_t, R_t  with 80% CI",
            "beta_t  (time-varying transmission)",
            "R_eff(t) = beta_t * S_t / (N * gamma)",
        ],
        "description": "Minimal 3-compartment model. Single data channel (cases).",
        "use_case":    "Rapid proof-of-concept; influenza without severity data.",
    },
    "SEIR": {
        "class":      SEIRModel,
        "states":     ["S", "E", "I", "R"],
        "log_rw":     ["beta_t"],
        "exogenous":  [],
        "obs_channels": ["cases"],
        "outputs": [
            "S_t, E_t, I_t, R_t  with 80% CI",
            "beta_t  (time-varying transmission)",
            "R_eff(t) = beta_t * S_t / (N * gamma)",
            "Incubation lag via E compartment",
        ],
        "description": "4-compartment SEIR with exposed (latent) period.",
        "use_case":    "COVID-19 early waves; any disease with incubation period.",
    },
    "SEIRD": {
        "class":      SEIRDModel,
        "states":     ["S", "E", "I", "R", "D"],
        "log_rw":     ["beta_t", "delta_t"],
        "exogenous":  [],
        "obs_channels": ["cases", "deaths"],
        "outputs": [
            "S_t, E_t, I_t, R_t, D_t  with 80% CI",
            "beta_t  (transmission)",
            "delta_t  (IFR — infection fatality rate, time-varying)",
            "R_eff(t) = beta_t * S_t / (N * (gamma + delta_t))",
            "IFR_t   = delta_t / (gamma + delta_t)",
        ],
        "description": "SEIRD adds disease-caused deaths with time-varying IFR.",
        "use_case":    "Variant-driven severity shifts; excess-mortality estimation.",
    },
    "SEIRV": {
        "class":      SEIRVModel,
        "states":     ["S", "E", "I", "R", "V"],
        "log_rw":     ["beta_t"],
        "exogenous":  ["nu_t  (daily vaccination rate)"],
        "obs_channels": ["cases"],
        "outputs": [
            "S_t, E_t, I_t, R_t, V_t  with 80% CI",
            "beta_t  (transmission)",
            "R_eff(t) = beta_t*[S_t+(1-eps)*V_t] / (N*gamma)",
            "Herd immunity threshold = 1 - 1/R0_t",
            "Vaccine-attributable incidence reduction",
        ],
        "description": "SEIR + vaccination compartment with waning immunity.",
        "use_case":    "Rollout impact; booster timing; waning effectiveness.",
    },
    "SEIARV": {
        "class":      SEIARVModel,
        "states":     ["S", "E", "A", "I", "R", "V"],
        "log_rw":     ["beta_t", "Q_C_t"],
        "exogenous":  ["nu_t  (vaccination rate)"],
        "obs_channels": ["cases"],
        "outputs": [
            "S_t, E_t, A_t, I_t, R_t, V_t  with 80% CI",
            "beta_t  (transmission)",
            "Q_C_t   (case detection probability — jointly estimated)",
            "R_eff(t)  accounting for asymptomatic transmitters",
            "True incidence = symptomatic + asymptomatic",
            "Under-reporting factor = 1/Q_C_t",
        ],
        "description": "Adds asymptomatic (A) compartment and dynamic detection Q_C_t.",
        "use_case":    "Under-reporting estimation; testing policy evaluation.",
    },
    "SEIRHD": {
        "class":      SEIRHDModel,
        "states":     ["S", "E", "I", "H", "R", "D"],
        "log_rw":     ["beta_t", "tau_i_t", "delta_h_t"],
        "exogenous":  [],
        "obs_channels": ["cases", "hosp", "deaths"],
        "outputs": [
            "S_t, E_t, I_t, H_t, R_t, D_t  with 80% CI",
            "beta_t     (transmission)",
            "tau_i_t    (hospitalisation rate from I, time-varying)",
            "delta_h_t  (in-hospital CFR, time-varying)",
            "R_eff(t) = beta_t * S_t / (N*(gamma_i+tau_i_t))",
            "Hospitalisation rate_t = tau_i_t/(gamma_i+tau_i_t)",
            "In-hospital CFR_t = delta_h_t/(gamma_h+delta_h_t)",
        ],
        "description": "3-channel model: cases + hospital occupancy + deaths.",
        "use_case":    "Hospital surge planning; severity surveillance.",
    },
    "SVEAIHCRD": {
        "class":      _sveaihcrd,   # lazy — avoids import at load time
        "states":     ["S", "V", "E", "A", "I", "H", "C", "R", "D"],
        "log_rw":     ["beta_t", "tau_i_t", "delta_h_t", "rho_c_t", "Q_C_t"],
        "exogenous":  ["nu_t  (vaccination rate)"],
        "obs_channels": ["cases", "icu", "hosp", "deaths"],
        "outputs": [
            "All 9 compartments with 80% CI",
            "5 time-varying parameters with posterior trajectories",
            "R_eff(t) with ICU and hospitalisation burden",
            "Optimal two-control intervention (NPI u_t + testing v_t)",
            "PF-MPC receding-horizon adaptive policy",
            "Pareto frontier: NPI vs testing cost",
        ],
        "description": "Full 9-compartment 5-parameter model. Germany COVID-19 validated.",
        "use_case":    "Complete national-level COVID-19 surveillance + control.",
    },
}


def get_model(name: str):
    """Return the EpiModel class for `name`."""
    if name not in MODEL_INFO:
        raise KeyError(f"Unknown model '{name}'. Available: {list(MODEL_INFO)}")
    cls = MODEL_INFO[name]["class"]
    return cls() if callable(cls) and not isinstance(cls, type) else cls


def list_models(verbose: bool = True) -> list[str]:
    """Print a summary table of all available models."""
    lines = [
        f"{'Model':<12} {'States':<6} {'Log-RW params':<25} {'Obs channels':<30} Description",
        "-" * 110,
    ]
    for name, info in MODEL_INFO.items():
        n_s   = len(info["states"])
        rw    = ", ".join(info["log_rw"])
        ch    = ", ".join(info["obs_channels"])
        lines.append(f"{name:<12} {n_s:<6} {rw:<25} {ch:<30} {info['description']}")
    if verbose:
        print("\n".join(lines))
    return list(MODEL_INFO.keys())


def model_outputs(name: str) -> None:
    """Print all outputs available for a given model."""
    if name not in MODEL_INFO:
        raise KeyError(f"Unknown model '{name}'")
    info = MODEL_INFO[name]
    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"{'='*60}")
    print(f"Description : {info['description']}")
    print(f"Use case    : {info['use_case']}")
    print(f"States      : {info['states']}")
    print(f"Log-RW      : {info['log_rw']}")
    print(f"Exogenous   : {info['exogenous'] or 'none'}")
    print(f"Obs channels: {info['obs_channels']}")
    print(f"\nOutputs:")
    for o in info["outputs"]:
        print(f"  • {o}")
    print()
