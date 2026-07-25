# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- Adopted the py-canon fleet template: reusable CI, docs, and release
  workflows, dependabot version updates with auto-merge for patch/minor
  bumps, ruff + pyright + pydoclint conformance, and pre-commit hooks.
- Migrated the build backend to hatchling with uv-dynamic-versioning;
  the package version is now derived from git tags.
- Moved license metadata to PEP 639 SPDX form.
- Hybrid planning with `strategy="percentile_split"` now raises
  `ValueError` from the planner as well when `provision_percentile` is
  unset, instead of relying solely on target validation.

## [0.3.0] - 2026-04-15 (unpublished)

### Added

- Hybrid provisioned + pay-as-you-go planning with premium paygo backup.

## [0.2.0] - 2026-03-06

### Added

- Streamlit app.

## [0.1.0] - 2026-03-04

### Added

- Initial release: request trace ingestion, capacity simulation, and
  throughput/latency-target planning for reserved LLM capacity.
