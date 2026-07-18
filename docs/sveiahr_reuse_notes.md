# Reuse of the existing SVEIAHR project

The existing manuscript provides a useful calibration convention: it separates fixed biological
rates from dynamic transmission/severity paths, uses a case reporting check near 0.709, and treats
hospitalization as an admissions-based proxy with an 8-day length of stay. These values are imported
in `config/germany_calibration.yaml` as initialization references only. They are not copied into the
new model as validated SVEAIHCRD posterior estimates because the compartment structures differ.

The existing project also confirms that ICU occupancy is from DIVI and vaccination inputs are derived
from OWID cumulative coverage. This provenance is retained in the integrated-panel documentation.
