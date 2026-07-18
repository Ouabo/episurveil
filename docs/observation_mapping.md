# Observation mapping

The surveillance likelihood distinguishes latent flows from reported measurements. Daily hospitalization admissions are mapped to an occupancy proxy through a geometric length-of-stay kernel (`occupancy_from_admissions`), while cases and deaths may be delayed with a discretized gamma reporting-delay kernel (`delayed_incidence`). These transformations prevent admissions-based RKI series from being treated as instantaneous compartment occupancy and make the observation model explicit.
