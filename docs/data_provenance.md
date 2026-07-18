# Data provenance

The initial release uses synthetic demonstration data generated locally. Configured candidate sources
are listed in `config/data_sources.yaml`; `scripts/download_data.py` downloads them only when
explicitly run and writes a JSON metadata record containing URL, retrieval time, licence, status, and
SHA-256 checksum. No external dataset is claimed as downloaded until that metadata reports
`downloaded`. Validation and preprocessing are separate steps, and failed downloads do not generate
fabricated data.
