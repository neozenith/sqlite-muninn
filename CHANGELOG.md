# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
## [Unreleased]

### Features

- Update library agentic skills
- Refactor rewrite of agent skills
- Update version stamping for marketplace manifests for skills

### Other

- Refactor viz (#27)

* refactor: start rewrite of viz demo

* stamp new rc version

* continued rewrite of viz

* Commit changes about viz navigation and spheres for embedding markers

* Add sub charts in navigation

* docs: refactor documentation

* docs: refactoring the viz/ kg visualisations

* feat: improvements to knowledge graph colouring, filterings, node and edge sizing and layout algorithm configs

* chore: push min-degree server-side, isolate community opacity, unify KG restyle

Server:
- `min_degree` is now a query param on /kg/{table}; pruned after BFS expansion so
  communities shrink their node_ids and member_count with the same filter, and
  empty communities are dropped. Echoes back via KGPayload.min_degree.
- seed_metric/max_depth validation surfaces as 400 with a clear detail message.
- `_GraphCache` keyed by (db_path, table, resolution) memoizes BC across
  requests that only tune top_n/max_depth/seed_metric.

Frontend:
- Community opacity now applies only to compound parents via the three
  per-component properties (background/border/text opacity) rather than
  element-level `opacity`, which cascades to children in cytoscape.
- Seven separate styling `useEffect`s collapsed into one unified restyle pass
  that runs on every filter/colour/size change. Visibility runs first,
  producing hidden-id sets that feed normalization-aware node-size / edge-
  thickness so hidden outliers don't compress the visible range.
- minDegree joins the pending-reload group alongside topN/maxDepth/seedMetric.
- Theme context split into theme-context.ts so react-refresh stays happy.
- `resolved` theme derived via useMemo rather than a setState-in-effect round-trip.

CI pass:
- viz: ruff, prettier, mypy, eslint, pytest, vitest, playwright all green.
- benchmarks/harness: mypy annotation fixes on pre-existing type-arg errors
  (dict[str, Any] in s3_mirror + kg_api_adapters) so `make ci-all` advances
  past typecheck.

## [0.3.3] - 2026-04-21

### Other

- Finally publish to both npm and pypi after resolving deployment issues

## [0.3.3-rc1] - 2026-04-21

### Bug Fixes

- Fix release pipeline

### Other

- Fix/release pipeline (#26)

* attempt to fix release pipeline fo v0.3.2

* fix: address deploy pipeline issues

* fix: relock uv.lock file

* docs: update changelog
- Bump version and look at prototyping RC versions
- Another try

## [0.3.1] - 2026-04-21

### Other

- Fix/deploy pipeline (#25)

* fix: update ccache action versions

* fix: updated test targets

* fix: Increase macos build timeouts

## [0.3.0] - 2026-04-21

### Other

- Update plan docs
- Refined plan
- Feat/benchmarks cloud support (#21)

* tidy quality gates on er spec and remove google colab make targets since it is WIP

* benchmarks: benchmarks harness adding cloud awareness to be able to delegate jobs to cloud compute

* benchmark: fix missing check to rely on .jsonl files instead of actual sqlite files (which get stupid big)
- Benchmarks-cloud-support part 2 (#22)

* wip: ER Example

* feat: add benchmarks/infra/ — config-driven EC2 benchmark runner

Single-script lifecycle for remote benchmarks:
- runner.py: setup/run/status/teardown with S3 heartbeat monitoring (15s polls)
- user_data.sh: parameterized instance bootstrap with heartbeat, ccache, SSM
- Config via YAML + env var overrides (no hardcoded values)
- Spot instances with on-demand fallback; auto-terminate hung instances (>180s stale)
- Instance self-selects benchmarks via `manifest --commands --missing --limit 1`

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* fix: em dash in SG description + IpProtocol parameter name

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* feat: CDK infrastructure for per-branch benchmark pipelines

Stacks:
- MuninnCleanup: Lambda + EventBridge weekly schedule to prune AMIs >7 days
  (manually invokable: aws lambda invoke --function-name MuninnAmiCleanup)
- MuninnBench-{branch}: SQS queue + DLQ + ASG (spot with on-demand fallback)
  + step scaling (queue depth → 0..N workers). Scales to zero when idle.

Workers pull benchmark IDs from SQS, run via harness CLI, upload results
to S3. Spot interruption handled by SQS visibility timeout (message
reappears for retry). Poison pills go to DLQ after 3 attempts.

Also adds worker_user_data.sh (SQS-based worker bootstrap).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* add learnings for ER example to drop pairwise usage as an option and focus on the cluster style LLM comparrison grammar

* docs: add benchmarks/infra/README.md with full architecture and usage guide

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* feat: add prime and submit subcommands to runner.py

- prime: launches on-demand instance, monitors cold start, creates AMI,
  updates config.yml with new ami_id
- submit: queries manifest for missing benchmarks, enqueues IDs to the
  branch's SQS queue (looked up via CloudFormation stack outputs)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* docs: replace ASCII diagrams with Mermaid in benchmarks/infra/README.md

Three diagrams: single instance mode, parallel workers (CDK), AMI lifecycle.
All validated via mmdc rendering.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* fix: prime-only mode skips benchmarks when limit=0, max_workers from context

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* fix: extend AMI waiter timeout to 30 minutes

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* wip erv2

* wip erv2

* feat: add Plotly Dash monitoring dashboard for benchmark deployments

Auto-refreshes every 15s showing: SQS queue depth, ASG worker count,
per-instance heartbeat status/phase, and event log timeline.

Usage: uv run benchmarks/infra/dashboard.py (opens at localhost:8050)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* fix: scaling policy uses visible+inflight messages, 10min cooldown before scale-in

The naive visible-only metric scaled in while workers were actively processing
(messages go invisible when pulled). Now uses a math expression summing both
visible and in-flight messages, requiring 10 consecutive zero-periods before
scaling in.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* feat: add time-series line chart to dashboard (queue depth + workers over time)

Dual y-axis: messages (left) and workers (right). Accumulates data points
in dcc.Store on each 15s refresh, keeps last 240 points (~1 hour window).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* feat: CloudWatch-backed time-series chart + ASG scaling events table

Chart now queries CloudWatch metrics (last 1 hour) instead of ephemeral
in-memory state. Full page refresh preserves all historical data.

Scaling Events table shows ASG activities classified as: SCALE OUT,
SCALE IN, SPOT RECLAIM, UNHEALTHY, TERMINATE — with instance IDs,
capacity changes, and timestamps from describe-scaling-activities.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* updated default values for string only er to improve ER before LLM borderline cases

* fix: cloud-init clean before shutdown so AMI-launched instances re-run user-data

Without this, cloud-init considers user-data "already ran" from the prime
boot and skips it on ASG-launched instances. Workers would boot, poll SQS
for 20s, find nothing (because user-data never ran the SQS worker loop),
and shut down — silently wasting every run.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* ER analysis of Fp/Fn cases as well as cross comparing embedding models

* fix: systemd service replaces cloud-init for worker boot execution

Cloud-init only runs user-data on first boot. AMIs baked from a primed
instance skip user-data on subsequent launches, causing workers to run
the old prime script (limit=0) instead of the SQS worker script.

Fix: prime installs a systemd oneshot service (muninn-worker.service)
that runs on every boot after network-online.target. It downloads
scripts/worker.sh from S3 and executes it. No cloud-init involvement.

runner.py submit now uploads the rendered worker script to S3 before
enqueuing benchmarks, so the systemd service always gets the latest
version with the correct SQS queue URL.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* feat: time range selector for dashboard chart (1h/3h/12h/1d/3d/7d)

Radio buttons switch CloudWatch query window. Period auto-scales:
1-3h = 1min, 3-24h = 5min, 1-7d = 1h granularity.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* revised er pipeline

* improving stage timing granularity

* fix: npm lock sync, exclude KG categories, add benchmarks/infra/CLAUDE.md

- npm/package-lock.json: regenerated to match @sqlite-muninn v0.3.0-alpha.1
- registry.py: KG categories excluded by default (BENCH_EXCLUDE_CATEGORIES env
  var to override). Prevents cloud workers from attempting unvalidated KG benchmarks.
- benchmarks/infra/CLAUDE.md: documents all gotchas (cloud-init, spot SIGKILL,
  SQS scaling, llama.cpp OOM, category exclusion, cross-region S3)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* version control claude.md files

* fix: WCAG AA compliant color palette for dashboard

All colors now use Tailwind v3 shades with verified contrast ratios:
- Text: slate-100 (#f1f5f9, 14.5:1), slate-300 (#cbd5e1, 9.1:1),
  slate-400 (#94a3b8, 5.6:1) on slate-900 background
- Accents: red-400, violet-400, amber-400, green-400 (all >5:1 on slate-800)
- Conditional rows: green-950, red-950, amber-950 with matching accent text
- Previously: #533483 on #0f3460 was ~1.5:1 (unreadable),
  #2d4a22 on #0f3460 was ~1.2:1 (invisible)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* updated charts for ER Example analysis

* fix: radio button text color + default time range to 3d

labelStyle color set to slate-300 (was inheriting black from browser default).
Default time range changed from 1h to 3d (72h) to show full deployment history.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* Add graph edge betweenness pruning of ERs

* add graph edge betweenness to benchmarks harness

* docs: cloud-enabled manifest pattern — gap analysis + agent-facing rule

Gap analysis at docs/plans/cloud_enabled_manifest_pattern.md identifies 5 gaps
between two manifest pattern implementations. Extracts the generalised 5-layer
pattern: permutation registry, status determination, manifest CLI, execution
lifecycle, cloud dispatch.

Rule at .claude/rules/python/helper_scripts/cloud_enabled_manifest_pattern.md
uses generic examples only (per agnostic rules). Target audience: AI agents.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* fast C implementation of ER pipeline

* escape JSON strings for entity results

* feat: Added muninn_label_groups Address issue with community and cluster naming stalling filling up on thinking tokens and not ending.

* feat: HNSW parameter sweep for VSS benchmarks (M, ef_construction, ef_search)

VSS Treatment now accepts tunable HNSW parameters. HNSW engines (muninn-hnsw,
vectorlite-hnsw) get a full sweep: M=[8,16,32,64] x ef_construction=[100,200]
x ef_search=[10,50,100,200,400]. Non-HNSW engines unchanged.

Total VSS permutations: 420 -> 3,486 (3,360 HNSW + 126 non-HNSW).
Backward compatible: existing results (default params) keep their original
permutation_id without suffix.

Parameter ranges based on research across hnswlib, pgvector, Weaviate, Qdrant,
Milvus, and academic benchmarks. M has the most impact on memory, ef_search on
recall/latency tradeoff, ef_construction on build quality.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* updates to session_demo

* fix cache init on sessions_demo

* feat: dashboard improvements — spot pricing, log viewer, metric fix, port 8060

- Running Instances card uses EC2 instance count (not ASG DesiredCapacity)
  to eliminate CloudWatch metric lag mismatch
- Workers table adds Lifecycle (spot/on-demand) and Spot Price columns
  via describe_spot_price_history per instance type + AZ
- CloudWatch Logs viewer: input instance ID, filter by ERROR/WARN/INFO,
  fetches last 60 minutes on demand (not auto-refresh)
- Port changed from 8050 to 8060

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* feat: dashboard — dual ASG series, cumulative spend, uptime/cost columns

Chart: separate In Service (solid) and Desired (dashed) lines fix the
metric card vs chart mismatch. Cumulative $ (cyan, 3rd y-axis) tracks
estimated spot spend over time from CloudWatch InServiceInstances.

Workers table: adds Uptime (hours), Cost (accumulated $), Spot $/hr.

Also fix: systemd TimeoutStartSec=infinity (was 7200 = 2hr, killed
workers mid-benchmark when processing many permutations sequentially).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* feat: --port CLI arg for dashboard

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* fix: prep sync from S3, circuit breaker, systemd timeout=infinity

Three failure mode fixes:

1. Prep sync: both user_data.sh and worker_user_data.sh now sync
   vectors, texts, and GGUF models from S3 during boot. Validates
   all 12 vector caches exist, warns on missing files.

2. Circuit breaker: worker stops after 3 consecutive benchmark
   failures. Prevents burning through the entire SQS queue when
   there's a systemic issue (missing prep data, broken extension).
   Phase log now reports "N run, M failed" for auditing.

3. Systemd timeout: TimeoutStartSec=infinity (was 7200=2hr).
   Workers processing many benchmarks sequentially were killed at
   the 2hr mark. Heartbeat + client-side hung detection handles
   liveness — systemd should not impose its own timeout.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* fix: --limit 0 should pass limit=0, not skip the limit filter

`if limit:` was falsy for 0, skipping the filter and enqueuing all
missing benchmarks. Changed to `if limit is not None:`.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* feat: on-demand pricing comparison in dashboard workers table

Adds OnDemand $/hr and Saving % columns. On-demand price fetched via
EC2 Pricing API (us-east-1), cached per instance type to avoid repeated
calls. Shows spot vs on-demand savings percentage per worker.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* fix: radio buttons and inputs persist across 15s auto-refresh

Added persistence=True, persistence_type='local' to time-range radio,
log-level radio, and log-instance-id input. User selections are stored
in browser localStorage and survive interval refreshes and page reloads.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* feat: split queue and workers into separate line charts

Queue chart: Visible, In Flight, Dead Letter (messages y-axis)
Workers chart: In Service, Desired, Cumulative $ (workers + cost y-axes)

Also fix: duplicate margin kwarg in workers_fig.update_layout.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* fix: enable ASG group metrics for CloudWatch (GroupInServiceInstances etc)

ASG group metrics are opt-in — not published by default. The workers
chart showed zeros because CloudWatch had no data. Added
group_metrics=[GroupMetrics.all()] to the CDK ASG construct.

Also enabled via CLI on the running ASG for immediate effect.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* ensure GHA ci workflow uses only makefile targets

* address ci formatting and linting issues

* feat: multi-select instance dropdown for CloudWatch logs viewer

Replaces free-text input with a dcc.Dropdown(multi=True) populated from
running ASG instances + recent CloudWatch log streams. Labels show
"(running)" or "(recent)" status. Logs display includes instance ID
prefix per line for multi-instance views.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* fix: install + configure CloudWatch agent for /muninn/benchmarks logs

The CloudWatch agent was lost during the systemd refactor — neither
user_data.sh nor worker_user_data.sh referenced it. Instances wrote
to /var/log/muninn/benchmark.log locally but nothing shipped to
CloudWatch Logs. The dashboard log viewer had no data to show.

Fix: user_data.sh installs the agent package (baked into AMI).
worker_user_data.sh writes the config and starts the agent on every
boot with the instance ID as the log stream name.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* fix: upload no-op worker script during prime to prevent SQS polling

The AMI's systemd service downloads worker.sh from S3 on every boot.
During prime, this caused the instance to start pulling benchmarks
from the SQS queue instead of just doing the cold start. Fix: prime
uploads a no-op script before launching. The real worker script is
uploaded by runner.py submit when benchmarks are enqueued.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* feat: prime auto-suspends ASG scaling, resumes after AMI creation

Prime lifecycle now: suspend ASG alarms + scale to 0 -> launch prime
instance -> create AMI -> resume ASG alarms. Prevents the ASG from
launching workers from the old AMI while priming.

Also adds _get_asg_name() helper to look up ASG from CloudFormation.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* fix: prime monitor treats shutting-down as normal stop transition

The EC2 state machine goes running -> shutting-down -> stopped when
shutdown -h is called with InstanceInitiatedShutdownBehavior=stop.
The prime monitor was catching the brief shutting-down state and
aborting with exit code 1/2. Now waits for full stop.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* fix: remove duplicate cmd_run that was embedded inside cmd_prime

cmd_prime was missing its closing return — the cmd_run function body
was inlined as trailing code inside cmd_prime. After prime completed
AMI creation, execution fell through into cmd_run which launched a
spot instance and exited with code 2 on shutting-down.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* fix sessions_demo embedding process to be consistent

* refactor sessions_demo builder to decouple from demo_builder

* fix: CloudWatch agent + SSM install moved outside cmake cold-start check

Both were nested inside 'if ! command -v cmake' which is skipped on
warm AMI boots. Moved to top level with their own idempotent guards
so they install on first prime regardless of cmake state.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>

* disable claude code on GHA from burning quota for now

* fix: node2vec_train() arg mismatch — remove weight col, fix order, use neg_samples/lr_init

Python treatment was passing 14 args; C function expects 13.
- Remove 'weight' column arg (C loads edges without weight column)
- Swap walk_length/num_walks to match C order (num_walks first)
- Replace min_count=1 with neg_samples=5 (C param name)
- Replace workers=4 (unused) with lr_init=0.025 (C param)

This unblocks all 54 node2vec benchmarks currently failing with
"wrong number of arguments to function node2vec_train()".

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

* fix: truncate embed texts to model's max_tokens context window

muninn_embed() (via llama.cpp) rejects texts exceeding the model's context
window with "text too long: N tokens exceeds context of 512". Other backends
(sentence-transformers) silently truncate, so we align behaviour in setup().

- Add max_tokens to EMBEDDING_MODELS: MiniLM/BGE-Large=512, NomicEmbed=8192
- Truncate doc and query texts to max_tokens*4 chars at setup time
  (~4 chars/token for English WordPiece gives safe headroom below the limit)

This unblocks all embed benchmarks on wealth-of-nations_n5000 that were
failing with 543-token texts against 512-token MiniLM.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

* fix: stop uploading SQLite DBs to S3, clean up local DBs after each benchmark

The SQLite workspace DB is a scratch file — only the JSONL metrics matter.
Uploading it wasted S3 bandwidth and kept large files (HNSW shadow tables
can be many MB) on the instance disk indefinitely, causing disk-full failures
after running hundreds of benchmarks.

- harness.py: remove mirror.sync_to_s3(db_path) — JSONL only
- worker_user_data.sh: rm -rf results/{permutation_id}/ after each success
  to keep disk usage bounded regardless of how many jobs a worker runs

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

* fix: add sqlite-lembed to benchmark dep group so workers can run lembed treatments

sqlite-lembed was in the docs group, not benchmark. Workers run
uv sync --group benchmark, so all embed_lembed+* benchmarks were failing
with: ModuleNotFoundError: No module named 'sqlite_lembed'

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

* feat: add diagnose subcommand to runner.py

Moves failure analysis from the standalone tmp/diagnose_workers.py into
the proper CLI. Uses CloudWatch Logs Insights for server-side aggregation
— one query replaces dozens of per-stream CLI calls.

  uv run benchmarks/infra/runner.py diagnose
  uv run benchmarks/infra/runner.py diagnose --minutes 120

Reports queue depth, ASG desired/active/hung, known failure types with
fix-commit hints, failed job IDs, and unknown exception patterns.
Queue/ASG/DLQ resolved from CloudFormation outputs — no hardcoded ARNs.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

* fix: update llama.cpp to support Gemma4 models

* fix: defer llama_cpp import — remove vectors from prep/__init__.py

vectors.py has a top-level `from llama_cpp import ...` which was being
loaded whenever any benchmarks.harness.prep.* submodule was imported
(Python executes __init__.py first). This caused `manifest` and `benchmark`
subcommands to crash with ModuleNotFoundError on machines where llama_cpp
is not installed (e.g., the local submit path).

VECTOR_PREP_TASKS/VectorPrepTask are not imported from the package namespace
anywhere — both cli.py and test_prep.py import directly from prep.vectors.
Removing the re-export from __init__.py breaks the chain without changing
any callsites.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

* fix: always shutdown in worker EXIT trap, not just at normal exit

_cleanup (the bash EXIT trap) deletes the heartbeat file but did not
call shutdown. If the script exited abnormally — set -e crash, unhandled
error, circuit breaker — the heartbeat would go stale but the EC2 instance
stayed alive burning cost with no worker process running.

Add `shutdown -h now` to _cleanup so the instance terminates regardless
of how the script exits (normal completion, crash, or spot SIGTERM).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

* docs: benchmark run resume notes — OOM fix + resumption checklist

Documents current state (2737/3886 done), the VSS ground-truth OOM bug,
all fixes applied this session, and the exact steps to resume.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

* fix: increase EBS from 20→40 GiB; add disk usage report after sync phases

20 GiB had near-zero headroom at peak (OS + build tools + llama.cpp
static libs + Python venv + S3-synced vectors/models ≈ 16-18 GiB).
Any large HNSW DB during a benchmark run would hit the limit.

40 GiB gives ~20 GiB breathing room at ~$0.08/GiB/month (negligible for
spot instances running a few hours each).

Also adds a disk usage report (df -h + per-directory du) after phase 04b
so the prime boot logs show the exact footprint before the AMI snapshot
is taken — making it easy to catch headroom issues early.

Applied to both runner.py prime instance and CDK bench_stack.py workers.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

* fix: add HHMM to AMI name to prevent same-day re-prime collision

AMI names must be unique per account/region. Using only YYYYMMDD meant
re-priming on the same day (e.g. after fixing a bad prime) would fail
with InvalidAMIName.Duplicate. Including HHMM makes each prime unique.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

* fix: VSS OOM + add disk report to prime user_data.sh

- vss.py: replace (M,N,dim) broadcast in _compute_ground_truth with
  the L2 identity expansion (q^2 + d^2 - 2*q@d^T), reducing peak
  allocation from 14+ GiB to ~40 MB for N=50000,dim=768
- user_data.sh: add disk usage report after 04b_sync_prep, matching
  the same block already in worker_user_data.sh (prime runs
  user_data.sh, not worker_user_data.sh)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

* address ci fixes

* docs: update resume_benchmarks.md with session 2 state

- Mark VSS OOM as fixed (bcbd309)
- Add new AMI ami-02fd844e73df018b4 and disk usage numbers
- Update infrastructure state table (ASG suspended, CDK LT v7)
- Replace resumption checklist: resume-processes + submit + set-desired
- Add note: prime updates config.yml only; CDK redeploy is a separate step

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>

* remove unneeded examples

* consolidate aggregate tables

* add packaging badges. remove cruft docs plans

* update docs

* chore: update llama.cpp submodule to latest

* consolidate development knowledge and targets for inner loop

* update sessions demo to latest schema changes

* apply ruff format to sessions_demo/cache.py

* ci: use make test-c in C-only build steps

The "Run C unit tests" step and the ASan+UBSan step both called
`make test`, which now cascades into test-c + test-python + test-js +
docs-build. On Linux/macOS build jobs that has two bad effects:

1. It runs `vitest` before `npm ci`, failing with "vitest: not found".
2. It doubles up with the separate `Python integration tests` and
   `TypeScript tests` steps later in the same job.

Scope each step to its flavour: `make test-c` for the C-only invocation
(and for the sanitized build), leaving `make test-python` / `make
test-js` as the later dedicated steps after their respective setups.

---------

Co-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
- Get the current set of features out to unblock some production testing and update docs later (#24)

## [0.3.0-alpha.2] - 2026-03-26

### Bug Fixes

- Update Claude GHA permissions and action versions (#10)

### Other

- Increment version for another alpha release

## [0.3.0-alpha.1] - 2026-03-26

### Bug Fixes

- Address CI memory leak not unloading models. Address CI build issues needing to propagate from main build to windows build script.
- Demo db builder fully functional
- Remove FTS5Adapter from kg benchmarks
- Removed FTS5Adapater from KG benchmarks
- Llm_extract example compares a few models performing the NER and RE benchmarked against GLiNER2 models and the honest speed comparrisons
- Refactored viz demo
- Notebook kernel CWD is examples/{name}/, not project root
- Add --cache-bust flag to Colab E2E test script
- Github Action CI Issues (#4)

### CI

- Set concurrency groups to cancel in flight builds when a newer commit is available and set timeouts to cap jobs that are hanging in CMake.
- Refactor sources and build targets into one centralised spot
- Refactor wasm build pipeline and rebuild the kg-demo database for wasm/ and viz/

### Documentation

- Update docs with benchmark results for VSS for 50k and 100K embeddings
- Update planning documents
- Huge refactor of benchmarking pipeline to consolidate duplicated code across benchmarking and analysis tasks
- Update the text embedding example README
- Start the embedding benchmark docs pages
- Ran all the ag-news benchmarks for embed category
- Update graph benchmakrs
- Update feature list in README and mkdocs index
- Update feature list again to include dbt-syntax graph selection
- Refactor mermaid diagrams to hires pngs. clean out old specs
- Finalised the kg-demo.db builder script
- Update demo builder plan
- Update plan for wasm+viz merger and demo_builder
- Add documentation about the logo tooling to remove the background

### Features

- Full refactor of benchmarking prep tasks
- Adding llama.cpp integration
- Refactored the vss benchmark pipeline to use GGUF models for embeddings to be consistent with the impending embed category of benchmarks
- Updated demo_builder and session_demo for narrowest context window for kg pipelines
- Benchmarks.sessions_demo to build incremental knowledge graph from claude code sessions files. Lots of speed tuning splitting tasks into fine grained steps to find bottlenecks, making models work offline without needing constant internet checks or unnecessary redownloads of models.
- Refactor sessions_demo and demo_builder to add GLiNER2 backend, and incremental UMAP for demo pipelines
- Added llama.cpp chat and enhanced the NER and RE tasks
- Improve demo builder build subcommands.
- Refactor llama_common out of llama_embed and llama_chat. Added muninn_tokenize_text, also improved the muninn_summarize
- Add Colab notebook generation and README badge enforcement
- Rename examples to {name}.py, add nbmake test targets
- Add 3-environment path resolution for Colab + E2E test script
- Updated suport of supervised and unsupervised NER, RE

### Other

- Update dev tooling script for logo image processing to spit out full sequential step explanation.
- Huge refactor of demo builder
- Add Claude Code GitHub Workflow (#1)

* "Claude PR Assistant workflow"

* "Claude Code Review workflow"
- Updated plan docs
- Use qwen3.5 in example
- Address CI issues

### Refactor

- Refactor benchmarks cli usage docs, add updated plannign docs for next phases.

## [0.2.0] - 2026-02-18

### Bug Fixes

- Update amalgamation script with new files
- Address windows amalgamation script and name collision in amalgamation source code

### Documentation

- Updated refinement of upcoming plan documents

### Features

- Feat (graph): Graph adjacency virtual table with lazy incremental rebuild
- Implemented dbt graph selector syntax tvf

## [0.1.0] - 2026-02-18

### Bug Fixes

- Improved visualisation hover text on embedding vis
- Sanitise fts query strings in demo visualisations

### Documentation

- Hardcode absolute URL instead of relative URL to example to bypass mkdocs link resolver

### Other

- V0.1.0 release

## [0.1.0-alpha.10] - 2026-02-17

### Bug Fixes

- Refactor npm deployment to generate platform specific package.json and refined the wasm/ and viz/ demo servers
- Address ci code formatting issues

## [0.1.0-alpha.9] - 2026-02-17

### Bug Fixes

- Update the publish.yml to update the package-lock.json automatically in publishing but attempt to try to pre-resolve during make version-stamp target

## [0.1.0-alpha.8] - 2026-02-17

### Bug Fixes

- Address more npm publishing bugs

## [0.1.0-alpha.7] - 2026-02-17

### Bug Fixes

- Address build and publish sequence and tsup prePublishOnly hook not having devDependencies available

## [0.1.0-alpha.6] - 2026-02-16

### Bug Fixes

- Iterating on github action publishing to npm with trusted publishing

## [0.1.0-alpha.5] - 2026-02-16

### Bug Fixes

- Deploying to sqlite-muninn npm org instead
- Need to specify --tag when publishing to npm

## [0.1.0-alpha.2] - 2026-02-16

### Bug Fixes

- Deploy multi-platform binaries to npm

### Documentation

- Fixed the linked logo image in the readme for pypi and npm

## [0.1.0-alpha.1] - 2026-02-16

### Bug Fixes

- Fix CI: use pysqlite3-binary for extension loading, install uv

The actions/setup-python@v5 Python 3.13 builds lack
PY_SQLITE_ENABLE_LOAD_EXTENSION, so enable_load_extension() is
unavailable. Use pysqlite3-binary as a drop-in replacement in CI.
Also install uv via astral-sh/setup-uv@v4 for package install tests,
and fix the persistence test's extension path to use build/muninn.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
- Fix CI: pysqlite3 fallback for ARM64/macOS, add build/ to npm binary search

pysqlite3-binary only publishes wheels for Linux x86_64. Fall back to
pysqlite3 (source compile) on ARM64 and macOS where no binary wheel
exists.

Also add build/ directory to getLoadablePath() search order since
make all outputs to build/muninn.{so,dylib,dll}, not the repo root.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Documentation

- Doc (benchmarks): Rebuilding the benchmarked dataset
- Doc (benchmark): Rebuilding the benchmark dataset

### Features

- Add WASM demo, overhaul viz frontend, refine CD plan
- Add publish.yml workflow for release automation
- Build platform wheels natively in CI with uv

### Other

- Initial commit
- Initial benchmarking results
- Updated benchmark metrics results
- More benchmark docs updates
- Updated docs benchmarks
- Update docs dataset URL reference
- Checkpoint planning documents
- Project rename to sqlite-muninn
- Add project logo
- Add graph community and centrality
- Updated planning documents
- Add ci and agent skills as well as python and nodejs wrappers
- Some house keeping
- Huge CI refactor
- Iterate on fixing CI
- Started works on visualisation tool and planning out KG benchmarks

### Refactor

- Refactored some more of the manifests architecture

<!-- generated by git-cliff -->
