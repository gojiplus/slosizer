# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.3.1] - 2026-08-17

### Changed

- Migrated from Hatchling and tag-derived build versions to py-canon's native
  `uv_build` release contract: explicit project metadata matched to the release
  tag.
- Replaced persistent synthetic CSV files and generated CSV results with
  explicit-schema Parquet.
- Made the Vertex model catalog the single source for available-model
  documentation.

### Fixed

- Reject hybrid cost and latency calculations when a trace has no measurable
  observation span instead of producing unbounded hourly rates or utilization.
- Support short synthetic horizons while preserving the seeded four-hour
  benchmark exactly.
- Fill null optional token telemetry with its documented defaults while still
  rejecting invalid text, negative values, and output limits below observed
  output.
- Reject empty planning traces and invalid output-token sources with clear
  errors.
- Plot slack results using the shortest available analysis window instead of
  requiring a one-second window.

## [0.3.0] - 2026-08-16

### Added

- Profit-aware reserved-capacity planning with hard or priced latency SLOs,
  auditable candidate frontiers, and cross-model scenario comparison.
- Versioned TOML capacity catalogs and a public loader for private provider
  profiles.
- Request, response model, service tier, request ID, and business-value fields
  in normalized traces.
- Effective-dated rate cards with provider, model, region, deployment, currency,
  and source metadata.
- Hybrid provisioned + pay-as-you-go planning with premium paygo backup.

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
- Hybrid paygo costing now prices uncached input, cached input, output, and
  separately reported thinking tokens. Nonzero capacity choices respect the
  provider minimum, and zero provisioned units cannot bypass a latency SLO.
- Vertex profiles now load from a reviewed catalog checked on 2026-08-15 and
  cover the current documented text-model lineup.
- Time buckets now retain requests that arrive exactly on the last observed
  boundary and compute empty-bucket overflow without divide-by-zero warnings.
- Economic comparisons now reject mixed currencies and non-finite prices or
  business inputs. All planners use the same provider purchase grid, and
  reported latency quantiles use the same empirical convention as SLO
  qualification.

## [0.2.0] - 2026-03-06

### Added

- Streamlit app.

## [0.1.0] - 2026-03-04

### Added

- Initial release: request trace ingestion, capacity simulation, and
  throughput/latency-target planning for reserved LLM capacity.
