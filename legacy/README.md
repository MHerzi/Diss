# Legacy implementation

This directory contains the historical dissertation code, data snapshots and
the original `MainFile.m`. It is intentionally excluded from the canonical
MATLAB Project path.

Regression tests add this directory explicitly when comparing modern kernels
with legacy formulas. New model implementations must be added to `src/+diss`,
not here.

For an isolated historical reproduction session, call `setupLegacyPath` and
retain its returned cleanup object. Closing or clearing that object restores
the previous MATLAB path.
