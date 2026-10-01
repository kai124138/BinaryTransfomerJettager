# Code compatibility, October 1, 2026

The compatibility branch adds the existing research jet-pT weighting implementation
to the public pipeline. Weighting stays disabled by default, and validation and
checkpoint selection remain unweighted. All 38 historical model configurations and
62 public model configurations retain their exact bytes. Historical generators use
a separate namespace, and compatibility wrappers preserve the earlier entry points.
See [the interface notes](../../code/hgq2/COMPATIBILITY.md).

The completed local CPU preflight records **217 passing comparisons**:

| Engineering check | Passed |
| --- | ---: |
| Executable interface and behavior contracts | 21 |
| Original-versus-candidate config builds | 100 |
| Original-versus-candidate checkpoint reload combinations | 96 |

The reload checks use deterministic synthetic probes. Their maximum observed output
difference is zero, within the required absolute tolerance of `1e-7`. These are
software compatibility results, not accuracy measurements. Five regression tests
also check the preflight runner's failure handling, output isolation and provenance.
The repository validator and whitespace checks pass.

Reload coverage accounts for 206 checkpoint paths, 94 config/seed name pairs and
29 research configurations after content and preprocessing deduplication. Another
74 checkpoint paths lack attributable metadata. Nine research configurations and
the exact renamed public configuration identities lack established historical
coverage. Short config hashes provide lookup associations; they do not prove the
original training source or complete historical provenance.

Historical metric reproduction and calibrated-width remeasurement remain pending.
This branch does not establish full historical compatibility or retire uncovered
entry points. Frozen Engram, Chang and Delta implementations remain authoritative
for their campaigns; they are not integrated by this change. Current constrained
candidates still require their own successful hardware validation.

The lab evidence is `campaigns/2026-09-26-code-line-merge/PREFLIGHT.md` and
`evidence/preflight_v5.json`. Earlier attempts are explicitly superseded or invalid.
The validated candidate manifest SHA-256 is
`dfd1f68324dfa4f2f8eb68f8da3875d7209d89c8bb7d43848201abd260dc9de0`.
The [training-results record](RESULTS_AND_NEXT_STEPS_20261001.md) remains unchanged.
