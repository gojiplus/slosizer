# API Reference

Use `plan_hybrid_capacity()` for the normal fixed-demand case. It minimizes the cost of provisioned capacity and paygo overflow subject to an optional hard SLO. Because the trace fixes gross value, this choice also maximizes profit without requiring a business-value estimate.

Use `plan_profit_capacity()` only when you need absolute expected profit or a priced SLO. Use `compare_profit_scenarios()` only when separate model forecasts justify different request traces or request values.

## Schema

Core data structures for capacity planning.

```{eval-rst}
.. automodule:: slosizer.schema
   :members:
```

## Ingestion

Request trace parsing and normalization.

```{eval-rst}
.. automodule:: slosizer.ingest
   :members:
```

## Simulation

Capacity simulation for queue-based latency modeling.

```{eval-rst}
.. automodule:: slosizer.simulation
   :members:
```

## Planning

Capacity planning algorithms.

```{eval-rst}
.. automodule:: slosizer.planning
   :members:
```

## Economics

Optional absolute-profit, priced-SLO, and cross-model analysis.

```{eval-rst}
.. automodule:: slosizer.economics
   :members:
```

## Catalogs

Versioned provider capacity facts.

```{eval-rst}
.. automodule:: slosizer.catalog
   :members:
```

## Plotting

Visualization functions.

```{eval-rst}
.. automodule:: slosizer.plotting
   :members:
```

## Synthetic Workloads

Synthetic workload generation for testing.

```{eval-rst}
.. automodule:: slosizer.synthetic
   :members:
```

## Provider Adapters

### Vertex AI

```{eval-rst}
.. automodule:: slosizer.providers.vertex
   :members:
```

### Azure OpenAI

```{eval-rst}
.. automodule:: slosizer.providers.azure
   :members:
```
