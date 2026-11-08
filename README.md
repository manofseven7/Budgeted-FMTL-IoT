# Budgeted federated multi-task IoT learning: current review release

Corresponding manuscript: **IoT-74649-2026**, repair protocol **v3**.
This package supplies the current executable study for peer review and GitHub
upload. It is a simulation study; no physical device-energy claim is made.

## Current evidence

- 276 validation-only trials: twelve method-specific configurations per dataset,
  100 rounds each, calibration seed 707, no test arrays in tuning files.
- 115 selected main fits, 15 scheduled-TopK controls and 30 matched sensor
  architecture controls: 300 rounds each; seeds 11/22/33/44/55.
- All 160 current final outcomes reconstruct from saved states; 22 tests pass.
- 600 heterogeneous-cost binding-budget cases pass feasibility checks and are
  compared with exact finite-subset optima.
- All completed historical runs are preserved; none is retrained.

The selected proposed configuration uses rate 0.1, two local steps per task,
rho 0.3, overlap threshold sigma 0 and participation-deficit weight lambda 0.25.
It reduces model-update uplink by 10.02%,3.20%,5.09% versus the scheduled-TopK
control on UCI/CWRU/Intel. Complete logical-exchange savings are smaller:
4.64%,1.42%,2.21%. Classification quality is a trade-off, not a dominance claim.
Jain participation indices exceed 0.9997.

Every sensor comparator now uses the same fixed current-reading offset and
zero-head-initialized residual predictor. This repairs the previous absolute-tanh
failure. Current errors are close to local persistence; no practical forecasting
superiority is claimed. The matched architecture control reuses all earlier
absolute-model outputs and fits only missing residual counterparts.

## Install

Python 3.11+; the actual environment is Python 3.12.14 with versions in
`requirements-lock.txt` and `results/supplement_analysis/environment.json`.

```bash
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s tests -v
```

## Data

Raw/processed datasets are excluded from this GitHub archive. The separately
supplied processed-data checkpoint preserves NPZ identities for exact resume.
Original providers:

- UCI HAR: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- CWRU: https://engineering.case.edu/bearingdatacenter/download-data-file
- Intel Lab: https://db.csail.mit.edu/labdata/labdata.html

```bash
OPENBLAS_NUM_THREADS=1 python scripts/download_public.py
OPENBLAS_NUM_THREADS=1 python -m src.preprocess_public --datasets uci_har cwru
OPENBLAS_NUM_THREADS=1 python -m src.preprocess_intel_v2
OPENBLAS_NUM_THREADS=1 python scripts/prepare_repair.py
```

Official downloads are container-validated with source SHA-256 records. UCI inner
validation uses five training subjects; CWRU uses training load 2 for validation,
with entire recordings disjoint; final fits use the complete declared training
split. Inner scalers use inner training only. Intel uses true timestamps, purges
boundary-crossing pairs and retains training-only scaling. Its predictors use
the previous gap, never the future gap. Final test arrays are absent from the
tuning archives. The residual error equals absolute forecasting error under the
fixed offset. Provider terms apply; Intel contributors are Peter Bodik, Wei Hong,
Carlos Guestrin, Sam Madden, Mark Paskin and Romain Thibaux.

## Restore completed evidence before analyzing or resuming

All small raw logs/states are bundled losslessly for GitHub upload. Restore all
three archives so completed experiments are never repeated:

```bash
bash scripts/unpack_results.sh
tar --no-same-owner -xzf results/archive/supplementary_runs.tar.gz
tar --no-same-owner -xzf results/archive/repair_v3_runs.tar.gz
```

Regenerate current analysis without training:

```bash
OPENBLAS_NUM_THREADS=1 python scripts/analyze_repair.py
OPENBLAS_NUM_THREADS=1 python scripts/validate_architecture_control.py
OPENBLAS_NUM_THREADS=1 python scripts/budget_stress.py
```

Only if configurations are genuinely missing:

```bash
python scripts/run_repair.py
python scripts/run_architecture_control.py
```

The launchers use four CPU processes and one BLAS thread each. Source/config/data
hashes reuse matching completed outputs and reject mismatches. Do not delete a
completed manifest to bypass a mismatch. Restore the recorded processed-data
checkpoint when NPZ identities differ; investigate changes rather than silently
overwriting results. The original/earlier runners are retained for history and
are not the entry points for the current main tables.

## Method and accounting boundaries

`src/repair.py` is the current primary runner; SCAFFOLD dispatches to
`src/scaffold_weighted.py` so its population control mean matches sample weights; `src/core.py` preserves the
mathematical sparse-update/scheduler contract. `src/experiment.py` and
`src/supplementary.py` preserve the earlier implementation history.

SCAFFOLD uses a sample-weighted population control mean and fixed round-start server controls and applies completed deltas
after the round. Model uplink, auxiliary/control uplink and logical downlink are
separate. Proposed norm reports, Power-of-Choice losses, IFCA cluster identifiers
and SCAFFOLD client/server controls are counted. Sparse frames include indices
and metadata. Transport, encryption, retransmissions and broadcast efficiencies
are not measured. Fixed-size TopK can transmit selected zeros; no optimal
serializer claim is made. The primary budget is nonbinding; controlled
heterogeneous-cost cases separately demonstrate binding-budget behavior.

The Laplacian method is explicitly a surrogate. FedRep/IFCA are stated two-task
adaptations, not official benchmark-score reproductions. Equal trial/round tuning
budgets do not imply equal local FLOPs or globally optimal settings. Proposed
uses two local steps; some independently tuned methods choose four. The direct
scheduled-TopK control holds proposed's optimizer/scheduler settings fixed.
Freezing is not an active primary contribution; archived sensitivity results
are diagnostic only.

## Evidence provenance and statistical scope

Current tables and checks are in `results/repair/analysis/`: `summary.csv`,
`paired_holm.csv`, `validation.json`, `architecture_control_validation.json`,
`architecture_control_summary.csv`, `persistence.json` and
`reviewer_25_matrix.csv`. Current selected settings are in
`results/repair/selected.json`; all trial/final plans are in `configs/`.

The 345 original full runs and 100 earlier supplementary runs are preserved.
Epoch-counter-based Intel outcomes are historical only. Timestamp-v2 absolute
forecasting is a separate historical diagnostic and supplies the matched
architecture-control reference. Neither is mixed into current primary scores.
Earlier benchmark test diagnostics were inspected during development; current
choices use validation only, but results remain exploratory. Five-seed intervals
are marginal, and Holm families are comparator-by-two-task within each dataset.
Neither non-significance nor similar means establishes equivalence.

## Public archive and review attachment

Verified historical V1: https://doi.org/10.5281/zenodo.22779286

Concept DOI: https://doi.org/10.5281/zenodo.22779285

Source repository: https://github.com/manofseven7/Energy-Efficient-Federated-Learning

The checksum/identity audit is `results/posthoc/zenodo_verification.json`.
V1 predates the repair and is not falsely described as containing these additions.
This complete current package is directly supplied to reviewers as
`Supplementary_Reproducibility.zip`, identical to the GitHub upload ZIP.
It can be deposited as a subsequent public version; no new DOI is invented.

The audit additionally preserves 36 superseded uniform-control tuning trials and15 fits. They are excluded from current tables. `scripts/refine_scaffold_weighting.py` documents the correction; 22 tests include an unequal-client-size control-mean identity check.

Distribution: the GitHub archive includes exact prediction models, heads, routing and all logs. Large per-client SCAFFOLD optimizer-control caches are retained losslessly in the separate processed-data/full-audit checkpoint, at their original `results/repair/scaffold_weighted/.../final_state.npz` paths. Extracting that checkpoint overlays the full states and enables the population-control identity audit; prediction reconstruction does not require these caches. No original state was deleted.

Audit note: one nonselected calibration checkpoint had a truncated ZIP directory. Complete prediction/server-control arrays were recovered with local-record CRC checks and reproduced its logged validation scores exactly. The original bytes and recovery report are preserved under `results/repair/checkpoint_recovery/`. No missing optimizer-cache values were fabricated and no training was repeated.

For archived summary tables/earlier diagnostic analyzers only, restore:

```bash
tar --no-same-owner -xzf results/archive/historical_summary_tables.tar.gz
```

## Candidate-downlink correction (8 November 2026)

The current compressor uses cosine on complete compensated vectors (shared
parameters and zero-padded disjoint task heads). The server applies equal task
weights of one half, including each head. TopK uses a complete deterministic
lexsort, not a partial sort. These are the executed training semantics.

Power-of-Choice delivers the current model to all 48 candidate clients before
collecting losses; only 24 train. Future runs count these deliveries directly.
Run `python scripts/reconcile_communication.py` after restoring the result
archives to reconcile historical current-cohort counters without training. The
operation is idempotent. All 56 affected current records (36 tuning, 15 primary,
5 architecture controls) have been reconciled in this release. Original records
are in `provenance/poc_accounting_before.tar.gz`; the correction ledger is
`results/repair/analysis/communication_correction.json`. Existing training source
hashes remain unchanged because they identify the code that produced each fit.
The registered source compatibility pairs allow accounting-only cache reuse at
identical fit settings; other changes still reject the cache.

The 22 tests include an old/new PoC toy run with exactly equal model-state
arrays, corrected candidate deliveries, cache compatibility and rejection of a
changed learning rate. No completed benchmark fit was repeated for this fix.
Logical unicast bytes exclude transport headers, encryption, retransmissions
and physical broadcast behavior. The old public V1 DOI is historical; the
attached current source/results archive is the version for the present tables.
