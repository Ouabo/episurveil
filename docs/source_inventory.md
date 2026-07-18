# Source inventory

Inventory generated from the Desktop parent tree. Existing files are preserved and are not copied
into this project unless explicitly adapted.

| Source | Type | Reusable material | Use status |
|---|---|---|---|
| `Bayesian_SVEIAHR_Sequential_Project/sequential_sveiahr_project_note.tex` | LaTeX | SVEIAHR compartments, ODE properties, particle filtering, German validation | Adapted conceptually; not copied |
| `Bayesian_SVEIAHR_Sequential_Project/real_covid_germany_validation.py` | Python | Multi-channel surveillance preprocessing and particle-filter diagnostics | Adaptation planned |
| `Article_Multi_Patch_and_Filtering/.../Epidemic_MultiPatch_Filter_v2.tex` | LaTeX | Multipatch coupling and filtering ideas | Mathematical cross-check |
| `Articles/Stochastic_Epidemics/...` | LaTeX | Stochastic epidemic and partial-observation formulations | Literature/source review |
| `Epidemic_Filter_Project` | Mixed | Existing filtering experiments | Inspect before reuse |

The parent tree contains multiple overlapping SVEIAHR, SEAIHCRD, multipatch, and filtering
formulations. The present platform uses a new modular SVEAIHCRD specification and records the
earlier SVEIAHR work as related material. No parent file is overwritten, renamed, moved, or deleted.
Claims and numerical results are not imported unless reproduced by this project.
