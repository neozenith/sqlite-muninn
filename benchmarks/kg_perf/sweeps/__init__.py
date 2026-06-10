"""kg_perf sweep scripts.

Each module here drives a parametrized run across a range of corpus
sizes / workloads and emits a derived artifact (chart, CSV, JSONL) for
post-hoc analysis. Sweeps are meant to be invoked via
``python -m benchmarks.kg_perf.sweeps.<name>`` and are independent of
the steady-state ``time_one`` benchmark loop.
"""
