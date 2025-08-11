## [0.2] - 2025-08-11
### Changed
- Removed legacy re-export shims from pre-modularization.
- All runners and scripts now import directly from:
  - `src/features/*`
  - `src/models/*`
  - `src/train/*`
### Notes
- No functional changes expected. Metrics should remain stable.
