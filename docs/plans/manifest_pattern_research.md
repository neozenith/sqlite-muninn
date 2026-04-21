# Manifest Pattern Research: Prior Art for Batch Computation

Research into external patterns, tools, and prior art for the "manifest pattern" — where a system defines N permutations of work, tracks which are complete, and executes the missing ones.

**Date:** 2026-03-28

---

## Table of Contents

1. [AWS Batch Array Jobs](#1-aws-batch-array-jobs)
2. [Workflow Managers (Snakemake, Nextflow, Luigi, Airflow)](#2-workflow-managers)
3. [GNU Make Pattern and Python Equivalents](#3-gnu-make-pattern-and-python-equivalents)
4. [Dask Delayed and Ray](#4-dask-delayed-and-ray)
5. [Hydra (Facebook)](#5-hydra-facebook)
6. [Experiment Tracking (MLflow, W&B, Sacred)](#6-experiment-tracking)
7. [Task Queue Patterns (Celery, RQ, Dramatiq)](#7-task-queue-patterns)
8. [Fan-Out/Fan-In (AWS Step Functions, SQS+ASG)](#8-fan-outfan-in-patterns)
9. [Prefect and Optuna](#9-prefect-and-optuna)
10. [Comparison Matrix](#10-comparison-matrix)
11. [Relevance to Our Manifest Pattern](#11-relevance-to-our-manifest-pattern)

---

## 1. AWS Batch Array Jobs

### How It Defines the Parameter Space

AWS Batch array jobs let you submit a single parent job with an `arraySize` (2 to 10,000). Each child job runs the same container image and job definition but receives a unique `AWS_BATCH_JOB_ARRAY_INDEX` environment variable (0 to N-1).

The canonical **manifest pattern**: store a manifest file (text file, S3 JSON, etc.) where each line/entry corresponds to an array index. The container reads its index, looks up the corresponding work item, and processes it.

```
# manifest.txt (stored in container or fetched from S3)
red       # index 0
orange    # index 1
yellow    # index 2
...
```

The container entrypoint converts `AWS_BATCH_JOB_ARRAY_INDEX` to a 1-based line number and reads the corresponding parameter.

### How It Determines Completion

- Each child job has its own status (SUBMITTED, PENDING, RUNNABLE, STARTING, RUNNING, SUCCEEDED, FAILED).
- The parent job transitions to FAILED if **any** child fails (after all children complete).
- The parent transitions to SUCCEEDED only if **all** children succeed.
- No built-in "resume from where you left off" — you must resubmit the entire array or build custom logic to filter completed items.

### Retry/Failure Handling

- `attemptDurationSeconds` timeout per child job.
- Retry strategies configurable per job definition.
- **SEQUENTIAL** dependency: child N waits for child N-1.
- **N_TO_N** dependency: index-matched dependencies across array jobs (Job-B:5 waits for Job-A:5).

### Cloud Offloading

Native — this IS the cloud execution model.

### Cost/Complexity

- **Pro:** Zero infrastructure to manage. Submit one API call, get up to 10,000 parallel jobs.
- **Pro:** No SQS, no ASG, no worker scripts. AWS manages scheduling, placement, retries.
- **Con:** Requires Docker/ECR image. Not trivial for projects with C extensions + vendored deps.
- **Con:** No incremental resume — if 500/1000 jobs succeed and 500 fail, you must handle re-submission yourself.
- **Con:** Cold start latency: Fargate launch ~30-60s, EC2 launch ~2-5min.

### URLs

- [Array jobs - AWS Batch](https://docs.aws.amazon.com/batch/latest/userguide/array_jobs.html)
- [Example of an array job workflow](https://docs.aws.amazon.com/batch/latest/userguide/example_array_job.html)
- [Use the array job index to control job differentiation](https://docs.aws.amazon.com/batch/latest/userguide/array_index_example.html)
- [SubmitJob API Reference](https://docs.aws.amazon.com/batch/latest/APIReference/API_SubmitJob.html)
- [How to Use AWS Batch Array Jobs for Parallel Processing](https://oneuptime.com/blog/post/2026-02-12-use-aws-batch-array-jobs-for-parallel-processing/view)
- [Job definition parameters](https://docs.aws.amazon.com/batch/latest/userguide/job_definition_parameters.html)
- [AWS Batch - Fan Wange Econ](https://fanwangecon.github.io/Py4Econ/aws/batch/htmlpdfr/fs_aws_batch.html)

---

## 2. Workflow Managers

### Snakemake

**Parameter space:** Rules with wildcards. Each rule defines input/output file patterns. The DAG is derived by instantiating rules for each set of needed input files.

**Completion detection:** File timestamp comparison (mtime). If output is newer than all inputs, the rule is skipped. `--rerun-triggers=mtime` controls this. Snakemake also supports checkpoints for dynamic re-evaluation of the DAG after a rule completes.

**Resume semantics:** Re-running Snakemake automatically skips completed rules whose outputs exist and are newer than inputs. This is the "Make model" — file existence IS completion.

**Retries:** Configurable per rule. `--retries N` flag.

**Cloud backends:** Kubernetes, Google Cloud Life Sciences, AWS (via plugins), Slurm, LSF, PBS.

**Cost/complexity:** Low barrier for bioinformatics-style file-to-file pipelines. Overkill for simple parameter sweeps. The file-timestamp model breaks when outputs are database rows, not files.

### Nextflow

**Parameter space:** Channels feed data into processes. Each process instance is a task. The DAG is implicit from channel connections.

**Completion detection:** Hash-based caching. The task hash is computed from: session ID, task name, container image, environment, task inputs, task script, global variables, stub run status. Before execution, Nextflow checks if the cache contains a matching hash AND the outputs exist in the work directory.

**Cache modes:**
- **Standard:** Full path + modification timestamp + file size.
- **Lenient:** Path + size only (ignores timestamps, useful for NFS).

**Resume:** `-resume` flag. Can target a specific previous run by ID. Cloud cache (`NXF_CLOUDCACHE_PATH`) enables S3-backed caching for multi-reader/writer scenarios.

**Cloud backends:** AWS Batch, Google Cloud Batch, Azure Batch, Kubernetes, Slurm, PBS, LSF. First-class cloud support.

**Cost/complexity:** More complex than Snakemake. Better for production pipelines. The hash-based approach is more robust than timestamps but requires preserving the `.nextflow/cache` directory and work directories.

### Luigi (Spotify)

**Parameter space:** Tasks are Python classes with typed parameters. Each unique parameter combination creates a unique task instance.

**Completion detection:** Target-based. Each task declares output Targets, and a Target must implement `exists()`. A task is complete IFF all its output Targets exist. This is a **backward** dependency resolution — Luigi walks backward from the requested task, checking existence at each level.

**Resume:** Automatic. If `target.exists()` returns True, the task is skipped. Simple and elegant.

**Retries:** Built-in retry with configurable count and delay.

**Cloud backends:** Limited. Hadoop/HDFS support built in. S3 and GCS targets available. No managed cloud execution — you run the scheduler yourself.

**Cost/complexity:** Simple conceptually. The Target abstraction is powerful — a Target can be a file, S3 object, database row, or anything with `exists()`. However, Luigi lacks advanced scheduling, monitoring, and cloud-native features. Largely superseded by Airflow/Prefect.

### Apache Airflow

**Parameter space:** DAGs defined in Python. Tasks are operators. Parameterization via `dag_run.conf`, Jinja templating, or dynamic task generation.

**Completion detection:** Task instance state machine with rich states: `none`, `scheduled`, `queued`, `running`, `success`, `failed`, `skipped`, `up_for_retry`, `upstream_failed`, `deferred`, `up_for_reschedule`, `removed`. A DAG Run succeeds only if all leaf nodes are `success` or `skipped`.

**Resume:** Manual. You can mark failed tasks as "clear" to re-run them. Or use `depends_on_past=True` for automatic dependency on previous DAG runs.

**Retries:** Per-task `retries` count with `retry_delay` and `retry_exponential_backoff`. Default retries configurable globally.

**Cloud backends:** KubernetesExecutor, CeleryExecutor, AWS MWAA (managed Airflow). First-class cloud support.

**Cost/complexity:** Heavy. Requires a metadata database, web server, scheduler. Overkill for batch parameter sweeps. Best for recurring ETL/ML pipelines.

### URLs

- [Nextflow vs Snakemake Comparison](https://tasrieit.com/blog/nextflow-vs-snakemake-comprehensive-comparison-of-workflow-management)
- [Workflow management systems in Bioinformatics](https://biomadeira.github.io/2022-10-25-workflow-management)
- [Nextflow Caching and Resuming](https://www.nextflow.io/docs/latest/cache-and-resume.html)
- [Nextflow Cache Training](https://training.nextflow.io/2.1/basic_training/cache_and_resume/)
- [Snakemake vs Nextflow](https://www.biostars.org/p/258436/)
- [Luigi Tasks Documentation](https://luigi.readthedocs.io/en/stable/tasks.html)
- [Luigi Workflows](https://luigi.readthedocs.io/en/stable/workflows.html)
- [Luigi GitHub](https://github.com/spotify/luigi)
- [Airflow Tasks Documentation](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/tasks.html)
- [Airflow DAG Runs](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dag-run.html)
- [Rerun Airflow DAGs](https://www.astronomer.io/docs/learn/rerunning-dags)
- [Workflow Managers in Data Science](https://marcsingleton.github.io/posts/workflow-managers-in-data-science-nextflow-and-snakemake/)

---

## 3. GNU Make Pattern and Python Equivalents

### GNU Make

**Parameter space:** Targets with prerequisites. Pattern rules (`%.o: %.c`) generate targets from wildcards. The dependency graph is a DAG.

**Completion detection:** File timestamps. A target is up-to-date if it exists and is newer than all prerequisites. This is the OG manifest pattern.

**Parallelism:** `make -j8` runs up to 8 independent recipes concurrently. Make topologically sorts the DAG and fires all ready jobs up to the job limit. No configuration needed — it just works.

**Retries:** None built-in. A failed recipe stops the build (or continues with `-k`).

**Cloud offloading:** None. Local execution only (though distcc/ccache provide distributed compilation).

### doit (pydoit) — Python Make

**Parameter space:** Tasks defined as Python dicts or functions. Each task has `file_dep`, `targets`, `actions`, and optional `task_dep`.

**Completion detection:** More flexible than Make:
- `file_dep` changes tracked via MD5 hash (not just mtime).
- Target existence checked.
- `uptodate` callables for custom logic.
- `result_dep` for checking if a dependency task's return value changed.
- State persisted in a local DB (SQLite by default), not just filesystem timestamps.

**Parallelism:** `-n N` flag for parallel execution.

**Key advantage over Make:** Dependencies are on **tasks**, not just files. A task without file outputs can still be tracked for up-to-date status via its `uptodate` callable or result hash.

**Cost/complexity:** Lightweight. Pure Python. Good for projects that want Make semantics without Makefiles.

### joblib.Memory — Function-Level Caching

Not a build system, but relevant: `joblib.Memory` provides transparent disk-caching of function return values based on input argument hashing. If a function is called with the same arguments, the cached result is returned from disk.

**How it works:** Hashes all input arguments. Checks if a cached result exists on disk for that hash. If yes, loads and returns it. If no, executes the function and persists the result.

**Relevant because:** This is the "manifest pattern" at the function level — the hash of inputs IS the manifest key, and disk presence IS the completion check.

### Parsl

A Python parallel scripting library that lets you define task graphs with decorators and execute them on various backends (local threads, Slurm, Kubernetes, AWS). Supports map/reduce patterns.

### URLs

- [GNU Make Parallel](https://www.gnu.org/software/make/manual/html_node/Parallel.html)
- [pydoit - Python Task Runner](https://pydoit.org/)
- [pydoit Dependencies](https://pydoit.org/dependencies.html)
- [pydoit Up-to-date Checks](https://pydoit.org/uptodate.html)
- [doit: the goodest python task-runner](https://www.bitecode.dev/p/doit-the-goodest-python-task-runner)
- [pydoit GitHub](https://github.com/pydoit/doit)
- [joblib.Memory Documentation](https://joblib.readthedocs.io/en/latest/memory.html)
- [joblib.Memory Basic Usage](https://joblib.readthedocs.io/en/latest/auto_examples/memory_basic_usage.html)
- [joblib Parallel](https://joblib.readthedocs.io/en/stable/parallel.html)
- [Checkpoint using joblib.Memory and joblib.Parallel](https://joblib.readthedocs.io/en/latest/auto_examples/nested_parallel_memory.html)

---

## 4. Dask Delayed and Ray

### Dask Delayed

**Parameter space:** Functions decorated with `@dask.delayed` build a task graph lazily. The graph is constructed by normal Python code — no DSL.

**Completion detection:** No built-in persistence across runs. Dask has "opportunistic caching" (in-memory only, not available with distributed scheduler). The `dask.cache.Cache` stores intermediate results in memory based on cost heuristics.

**Retries:** `client.compute(retries=3)` or `dask.annotate(retries=3)` for distributed scheduler.

**Cloud offloading:** Dask Distributed runs on any cluster. Dask Gateway for Kubernetes/cloud deployment. Coiled provides managed Dask clusters.

**Cost/complexity:** Moderate. Great for parallelizing existing Python code. Weak on persistence — no "resume from where you left off" across process restarts without external checkpointing.

### Ray

**Parameter space:** Functions decorated with `@ray.remote` become tasks. Ray Tune provides hyperparameter sweep functionality with search spaces.

**Completion detection:** Ray's object store tracks task results. **Lineage reconstruction** automatically recovers lost objects by re-executing the task that created them (follows the task graph backward).

**Retries:** Default 3 retries for non-actor tasks. `max_retries=-1` for infinite. `retry_exceptions=[ExceptionType]` for application-level retries. System failures (worker crashes) always trigger retry.

**Fault tolerance:** If an object is lost (node failure), Ray first checks other nodes for copies, then re-executes the creating task. Limitations: objects from `ray.put()` are not recoverable; owner process death is not recoverable.

**Cloud offloading:** Ray Clusters on AWS, GCP, Azure. Anyscale provides managed Ray. KubeRay for Kubernetes.

**Cost/complexity:** Heavy runtime. Best for long-running distributed applications. Overkill for simple batch sweeps.

### URLs

- [Dask Delayed Documentation](https://docs.dask.org/en/stable/delayed.html)
- [Dask Opportunistic Caching](https://docs.dask.org/en/stable/caching.html)
- [Custom Workloads with Dask Delayed](https://examples.dask.org/delayed.html)
- [Dask Futures Tutorial](https://tutorial.dask.org/05_futures.html)
- [Dask on Ray](https://docs.ray.io/en/latest/ray-more-libs/dask-on-ray.html)
- [Ray Fault Tolerance](https://docs.ray.io/en/latest/ray-core/fault-tolerance.html)
- [Ray Task Fault Tolerance](https://docs.ray.io/en/latest/ray-core/fault_tolerance/tasks.html)
- [Ray Object Fault Tolerance](https://docs.ray.io/en/latest/ray-core/fault_tolerance/objects.html)
- [Coiled: Choosing an AWS Batch Alternative](https://docs.coiled.io/blog/choosing-an-aws-batch-alternative.html)

---

## 5. Hydra (Facebook)

### How It Defines the Parameter Space

Hydra's `--multirun` (or `-m`) flag generates the Cartesian product of sweep parameters:

```bash
python my_app.py --multirun db=mysql,postgresql schema=warehouse,support,school
# Generates 2 x 3 = 6 jobs
```

Extended sweep syntax:
- `x=range(1,10)` — numeric ranges
- `schema=glob(*)` — file pattern matching
- `schema=glob(*,exclude=w*)` — filtered patterns

Config-based sweeps in YAML:
```yaml
hydra:
  sweeper:
    params:
      db: mysql,postgresql
      schema: warehouse,support,school
```

### How It Determines Completion

**It does not.** Hydra has no built-in completion tracking or resume capability. Each multi-run is independent. Output is organized into timestamped directories with per-job subdirectories, but there is no mechanism to skip already-completed configurations.

### Retries/Failures

No built-in retry. If a job fails, it fails. You must resubmit the entire sweep or handle it externally.

### Cloud Offloading

Hydra Launchers provide execution backends:
- **JobLib Launcher** — local parallel execution
- **Ray Launcher** — distributed execution via Ray
- **Submitit Launcher** — Slurm cluster submission
- **RQ Launcher** — Redis Queue

### Cost/Complexity

- **Pro:** Excellent configuration composition. YAML-based, hierarchical, overridable.
- **Pro:** Sweeper plugins (Optuna, Ax) for intelligent search beyond grid.
- **Con:** No completion tracking. No resume. Not a workflow manager.
- **Con:** Primarily a configuration framework with sweep capabilities bolted on.

### URLs

- [Hydra Multi-run](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/)
- [Hydra Configuring Experiments](https://hydra.cc/docs/patterns/configuring_experiments/)
- [Hydra Getting Started](https://hydra.cc/docs/intro/)
- [Hydra — A fresh look at configuration (PyTorch blog)](https://medium.com/pytorch/hydra-a-fresh-look-at-configuration-for-machine-learning-projects-50583186b710)
- [Hydra Multi-run Discussion](https://github.com/facebookresearch/hydra/discussions/2526)
- [Hydra Unleashing Multi-run](https://pub.aimind.so/python-hydra-unleashing-the-power-of-multi-run-b1d1aa996507)

---

## 6. Experiment Tracking

### MLflow

**Parameter space:** Runs within Experiments. Each run logs parameters, metrics, artifacts, and tags. Parent-child run relationships via `mlflow.start_run(nested=True)`.

**Completion detection:** Run status: `RUNNING`, `FINISHED`, `FAILED`, `KILLED`. Search API: `mlflow.search_runs(filter_string="status = 'FINISHED'")`.

**Resume:** No built-in resume of failed runs. You can query for missing parameter combinations and submit new runs.

**Cloud:** MLflow Tracking Server can be hosted anywhere. Databricks provides managed MLflow. Artifacts stored in S3/GCS/Azure Blob.

### Weights & Biases (W&B)

**Parameter space:** Sweep configs in YAML or dict. Strategies: `grid`, `random`, `bayes`. Sweep agents pull configurations from a central controller.

**Completion detection:** W&B tracks all runs with real-time sync. The sweep controller knows which configurations have been tried. Early termination via Hyperband algorithm kills poorly-performing runs.

**Resume:** Sweeps can be resumed — agents continue pulling untried configurations. Individual runs can be resumed with `wandb.init(resume="must", id="run_id")`.

**Cloud:** W&B Launch for managed compute. Integrates with Kubernetes, Slurm, cloud VMs.

### Sacred

**Parameter space:** Config scopes — functions decorated with `@ex.config` that turn local variables into configuration entries.

**Completion detection:** Observer events: `queued_event`, `started_event`, `heartbeat_event`, `completed_event`, `interrupted_event`, `failed_event`. MongoDB observer stores all experiment metadata with status.

**Resume:** No built-in sweep resume. You can query MongoDB for missing configurations.

**Cloud:** No managed execution. MongoDB is the only persistent backend worth using.

### URLs

- [MLflow Experiment Tracking](https://mlflow.org/docs/latest/ml/tracking/)
- [MLflow AI Platform](https://mlflow.org/classical-ml/experiment-tracking)
- [W&B Experiment Tracking](https://wandb.ai/site/experiment-tracking/)
- [W&B Sweeps](https://wandb.ai/site/sweeps/)
- [W&B Sweep Config Keys](https://docs.wandb.ai/models/sweeps/sweep-config-keys)
- [W&B Sweeps on Launch](https://docs.wandb.ai/guides/launch/sweeps-on-launch/)
- [Sacred Observers](https://sacred.readthedocs.io/en/stable/observers.html)
- [Sacred Collected Information](https://sacred.readthedocs.io/en/stable/collected_information.html)
- [Sacred GitHub](https://github.com/IDSIA/sacred)
- [MLflow vs W&B vs ZenML](https://www.zenml.io/blog/mlflow-vs-weights-and-biases)
- [W&B vs MLflow vs Neptune](https://neptune.ai/vs/wandb-mlflow)

---

## 7. Task Queue Patterns

### Celery

**Parameter space:** Not inherent — Celery is a task execution engine. You define tasks and dispatch them. Parameter sweeps require external orchestration to generate and submit tasks.

**Completion detection:** Task states: PENDING, STARTED, SUCCESS, FAILURE, RETRY, REVOKED. Result backend (Redis, RabbitMQ, database) stores results. `AsyncResult.status` for polling.

**Retries:** `@task(max_retries=5, autoretry_for=(Exception,), retry_backoff=True, retry_jitter=True)`. Exponential backoff with jitter. Max delay capped at 600s by default.

**DLQ:** `acks_late=True` — acknowledge after completion, not on receipt. Failed tasks after max retries can be routed to a dead letter queue.

**Cloud:** Workers can run anywhere. No managed cloud offering, but integrates with any broker (RabbitMQ, Redis, SQS).

### RQ (Redis Queue)

**Parameter space:** Manual. Enqueue jobs with `queue.enqueue(func, arg1, arg2)`.

**Completion detection:** Job states in Redis. `job.get_status()` returns `queued`, `started`, `finished`, `failed`, `deferred`, `scheduled`.

**Retries:** `Retry(max=3, interval=[10, 30, 60])` for configurable retry with delays.

**DLQ:** Failed jobs go to a `FailedJobRegistry`. Manual inspection and re-queue.

### Dramatiq

**Parameter space:** Manual dispatch like Celery.

**Completion detection:** Result backends for persisting outcomes.

**Retries:** Built-in automatic retries (unlike Celery's default). Messages acked only after processing completes (like `acks_late`). Delayed tasks queued separately and moved back when ready.

**DLQ:** Dead letter queue support via middleware.

**Key difference from Celery:** Tasks are only acknowledged when done processing. If a worker dies mid-task, the message returns to the queue automatically.

### URLs

- [Celery Tasks Documentation](https://docs.celeryq.dev/en/stable/userguide/tasks.html)
- [Choosing the Right Python Task Queue](https://judoscale.com/blog/choose-python-task-queue)
- [Dramatiq Motivation](https://dramatiq.io/motivation.html)
- [Celery vs Dramatiq](https://stackshare.io/stackups/celery-vs-dramatiq)
- [Async Task Patterns in Django](https://medium.com/@connect.hashblock/async-task-patterns-in-django-choosing-between-celery-dramatiq-and-rq-bb14339291fc)
- [How to Build Task Queues with Dramatiq](https://oneuptime.com/blog/post/2026-01-24-python-task-queues-dramatiq/view)
- [How to Build a Job Queue with Celery and Redis](https://oneuptime.com/blog/post/2025-01-06-python-celery-redis-job-queue/view)
- [Retrying Failed Celery Tasks](https://testdriven.io/blog/retrying-failed-celery-tasks/)
- [Celery Task Resilience](https://blog.gitguardian.com/celery-tasks-retries-errors/)
- [Advanced Celery: Idempotency, Retries & Error Handling](https://www.vintasoftware.com/blog/celery-wild-tips-and-tricks-run-async-tasks-real-world)

---

## 8. Fan-Out/Fan-In Patterns

### AWS Step Functions Map State

**Parameter space:** JSON array passed to the Map state. Each element spawns one sub-workflow execution.

**Inline Map:** Max 40 concurrent iterations. Good for small fan-outs.

**Distributed Map:** Up to 10,000 parallel sub-workflows. Can read items from S3 (CSV, JSON, S3 inventory). Each child is a full Step Functions execution.

**Completion detection:** Step Functions tracks each child execution status. The Map state completes when all children complete. `MaxConcurrency` controls parallelism.

**Retries:** Per-state retry with `IntervalSeconds`, `MaxAttempts`, `BackoffRate`. Catch blocks for error routing.

**Cost/complexity:**
- **Pro:** Fully serverless. No infrastructure. Built-in retry, error handling, visualization.
- **Con:** $0.025 per 1,000 state transitions. A 10,000-job sweep with 5 states each = $1.25. Adds up for large sweeps.
- **Con:** 25,000 execution history events limit per execution.

### SQS + ASG + Spot Instances

The canonical "embarrassingly parallel" pattern on AWS:

1. **Producer** enqueues N messages (one per permutation) into SQS.
2. **ASG** scales workers based on `ApproximateNumberOfMessagesVisible + ApproximateNumberOfMessagesNotVisible`.
3. **Workers** poll SQS, process one message, delete it on success.
4. **DLQ** catches messages that fail after `maxReceiveCount` attempts.
5. **Spot instances** reduce cost by 60-90%.

**Completion detection:** Message deleted from queue = complete. DLQ = failed. Queue empty + no in-flight = all done.

**Visibility timeout:** Must be longer than max processing time. If a worker dies before deleting the message, it becomes visible again after the timeout and is retried by another worker. This provides automatic retry on spot termination.

**Idempotency requirement:** Workers MUST be idempotent because SQS is "at least once" delivery.

**Cost/complexity:**
- **Pro:** Extremely cost-effective with Spot. ~$0.067/hr for t3.xlarge spot.
- **Pro:** Natural retry on failure (visibility timeout).
- **Pro:** Scale to zero when queue is empty.
- **Con:** More moving parts (SQS, ASG, IAM, CloudWatch alarms).
- **Con:** CloudWatch metrics lag on idle queues (up to 15 min to start reporting).
- **Con:** Scale-in requires careful alarm design (`visible + inflight`, 10-period evaluation).

### URLs

- [Fan out batch jobs with Map state](https://docs.aws.amazon.com/step-functions/latest/dg/sample-batch-fan-out.html)
- [New Step Functions Dynamic Parallelism](https://aws.amazon.com/blogs/aws/new-step-functions-support-for-dynamic-parallelism/)
- [Step Functions Dynamic Parallelism Explained](https://medium.com/swlh/step-functions-dynamic-parallelism-fan-out-explained-83f911d5990)
- [Distributed Map Best Practices](https://dev.to/aws-builders/step-functions-distributed-map-best-practices-for-large-scale-batch-workloads-55n2)
- [Fan-Out/Fan-In Serverlessly in 2024](https://theburningmonk.com/2024/08/whats-the-best-way-to-do-fan-out-fan-in-serverlessly-in-2024/)
- [Design Pattern for Highly Parallel Compute: Recursive Scaling with SQS](https://aws.amazon.com/blogs/architecture/design-pattern-for-highly-parallel-compute-recursive-scaling-with-amazon-sqs/)
- [Running Cost-Effective Queue Workers with SQS and Spot](https://aws.amazon.com/blogs/compute/running-cost-effective-queue-workers-with-amazon-sqs-and-amazon-ec2-spot-instances/)
- [Scaling ASG with Dynamic SQS Target](https://aws.amazon.com/blogs/compute/scaling-an-asg-using-target-tracking-with-a-dynamic-sqs-target/)
- [SQS Dead Letter Queues](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-dead-letter-queues.html)
- [SQS Visibility Timeout](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-visibility-timeout.html)
- [ec2-spot-labs SQS+ASG CloudFormation](https://github.com/awslabs/ec2-spot-labs/blob/master/sqs-ec2-spot-asg/sqs-ec2-spot-asg.yaml)
- [SQS+EC2 Graviton Spot (Well-Architected Workshop)](https://github.com/aws-samples/sqs-ec2-graviton-spot)
- [Cost-effective Batch Processing with EC2 Spot](https://aws.amazon.com/blogs/compute/cost-effective-batch-processing-with-amazon-ec2-spot/)
- [AWS Well-Architected SQS Best Practices Part 3](https://aws.amazon.com/blogs/compute/implementing-aws-well-architected-best-practices-for-amazon-sqs-part-3/)
- [AWS Batch Features](https://aws.amazon.com/batch/features/)
- [HPC Lens Batch Architecture](https://docs.aws.amazon.com/wellarchitected/latest/high-performance-computing-lens/batch-based-architecture.html)

---

## 9. Prefect and Optuna

### Prefect

**Parameter space:** Flows and tasks defined with decorators. Parameterized via function arguments.

**Completion detection:** Result persistence. Tasks with `persist_result=True` store serialized outputs. On re-run, if the result exists at the storage key, the task loads the cached result instead of re-executing. Cache policies (`cache_key_fn`, `cache_expiration`) control when cached results are valid.

**Storage backends:** Local filesystem (`~/.prefect/storage/`), S3, GCS, Azure Blob.

**Retries:** `@task(retries=3, retry_delay_seconds=10)`. Exponential backoff available.

**Cloud:** Prefect Cloud (managed). Prefect Server (self-hosted). Workers on any infrastructure.

**Key insight:** Prefect's caching is closest to the manifest pattern — the storage key IS the manifest entry, and existence of the persisted result IS the completion check. Reports 50-70% cost reduction in rerun-heavy pipelines.

### Optuna

**Parameter space:** Study with search space defined via `trial.suggest_*()`. Supports grid, random, TPE, CMA-ES, and custom samplers.

**Completion detection:** Trial states: RUNNING, COMPLETE, FAIL, PRUNED, WAITING. Stored in a backend (SQLite, MySQL, PostgreSQL). Studies are identified by name + storage URL.

**Resume:** `optuna.create_study(study_name="my_study", storage="sqlite:///study.db", load_if_exists=True)`. Resumes from where it left off. **Caveat:** interrupted trials remain in RUNNING state — they are never retried automatically and require manual cleanup.

**Cloud:** Optuna Dashboard for visualization. No managed execution — you run trials yourself. Optuna + Kubernetes via Katib.

### URLs

- [Prefect Results Documentation](https://docs.prefect.io/v3/advanced/results)
- [Prefect Caching and Persistence](https://docs.prefect.io/core/concepts/persistence)
- [Prefect Open Source](https://www.prefect.io/prefect/open-source)
- [Prefect GitHub](https://github.com/PrefectHQ/prefect)
- [Prefect DataOps Flows with Caching and Retries 2025](https://johal.in/prefect-dataops-flows-orchestration-with-caching-and-retries-2025/)
- [Optuna FAQ](https://optuna.readthedocs.io/en/stable/faq.html)
- [Optuna GitHub](https://github.com/optuna/optuna)
- [Optuna Study API](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html)

---

## 10. Comparison Matrix

| Tool | Parameter Space | Completion Check | Resume | Retries | Cloud Exec | Complexity |
|------|----------------|-----------------|--------|---------|------------|------------|
| **AWS Batch Array** | Array index (0..N) | Job status API | No (resubmit) | Per-job | Native | Medium |
| **Snakemake** | Wildcards + rules | File mtime | Automatic | Per-rule | Plugins | Low-Medium |
| **Nextflow** | Channels | Hash + file exists | `-resume` flag | Per-process | Native (AWS/GCP/K8s) | Medium |
| **Luigi** | Task params | `Target.exists()` | Automatic | Built-in | Limited | Low |
| **Airflow** | DAG + operators | Task state DB | Manual clear | Per-task | KubernetesExecutor | High |
| **GNU Make** | Targets + prereqs | File mtime | Automatic | None | None | Low |
| **doit** | Task dicts | MD5/mtime/custom | Automatic | None | None | Low |
| **Dask Delayed** | Python functions | In-memory only | No | `retries=N` | Dask clusters | Medium |
| **Ray** | `@ray.remote` | Object store + lineage | Lineage recon | `max_retries` | Ray clusters | High |
| **Hydra** | Sweep syntax | None | None | None | Launcher plugins | Low |
| **MLflow** | Run params | Run status API | No (query+resubmit) | None | Tracking server | Low-Medium |
| **W&B** | Sweep config | Sweep controller | Sweep resume | None | W&B Launch | Low-Medium |
| **Sacred** | Config scopes | Observer events | No | None | None | Low |
| **Celery** | Manual dispatch | Task state backend | No | `max_retries` + backoff | Any broker | Medium |
| **Dramatiq** | Manual dispatch | Result backend | No (auto re-queue on death) | Auto | Any broker | Low-Medium |
| **Prefect** | Flow/task params | Persisted results | Cache-based | Per-task | Prefect Cloud | Medium |
| **Optuna** | Search space | Trial state DB | `load_if_exists` | None (manual) | None | Low-Medium |
| **Step Functions Map** | JSON array | Execution status | Per-execution | Per-state | Native | Medium |
| **SQS + ASG** | Messages | Queue deletion | Visibility timeout | `maxReceiveCount` | Native | Medium-High |

---

## 11. Relevance to Our Manifest Pattern

### What We Already Have

The muninn benchmark harness implements a manifest pattern that combines elements from several of these systems:

1. **Registry** (`registry.py`): Enumerates all permutations (like Hydra's sweep or W&B's grid).
2. **JSONL result files** (`results/`): Completion check is file existence (like Make/Snakemake).
3. **`manifest --missing`**: Filters to incomplete work (like Luigi's `Target.exists()` backward check).
4. **SQS + ASG** (`runner.py` + CDK): Cloud offloading with spot instances, DLQ, visibility timeout retry.
5. **Idempotent workers**: Each benchmark is self-contained — rerunning produces the same JSONL.

### What We Could Learn From

**From Nextflow:** Hash-based caching instead of file-existence checks. A permutation's "done" status could be a content hash of its parameters, not just whether a JSONL file with the right name exists. This catches cases where parameters change but the filename doesn't.

**From Luigi:** The `Target.exists()` abstraction. Our completion check is currently "does `results/{permutation_id}.jsonl` exist?" — this could be generalized to `Target` objects that check S3, local files, or database rows.

**From Prefect:** Cache key functions. Instead of just checking file existence, hash the treatment parameters to create a cache key. If the result at that key exists and hasn't expired, skip.

**From AWS Batch Array Jobs:** The N_TO_N dependency pattern for multi-stage benchmarks. If we had a prep stage and a run stage, each permutation's run could depend on its specific prep.

**From doit:** Result-based dependencies (not just file dependencies). A treatment's "done" status could depend on the result of its prep treatment, not just file timestamps.

**From Dramatiq:** Ack-after-completion semantics. Our SQS workers already do this (delete message after benchmark completes), which is the right pattern for spot-interruptible work.

### Architecture Classification

Our system is best classified as a **"manifest-driven embarrassingly parallel batch executor with SQS fan-out"**. The closest prior art is:

1. **AWS Batch Array Jobs** — same concept (index -> work item), but we use SQS instead of Batch's built-in array indexing. Our approach is more flexible (workers self-select from manifest) but requires more infrastructure.

2. **Snakemake/Make** — same "skip if output exists" semantics. We use JSONL file existence where they use file timestamps.

3. **Luigi** — same backward-resolution pattern. Our `manifest --missing` is equivalent to Luigi walking the dependency tree and checking `Target.exists()`.

The key differentiator in our system is **worker self-selection**: workers run `manifest --commands --missing --limit 1` to pick the next incomplete permutation. This is more resilient to spot interruption than AWS Batch's fixed index assignment — if a worker dies, the permutation stays "missing" and the next worker picks it up.

### Tools NOT Worth Adopting

- **Airflow/Prefect:** Too heavy for our use case. We don't need a scheduler, web UI, or metadata database. Our permutations are independent, not a DAG.
- **Ray/Dask:** Our tasks are IO-bound (SQLite + model inference), not CPU-bound Python. The overhead of a distributed runtime doesn't help.
- **Hydra:** Our parameter space is already well-defined in `registry.py`. Hydra's value is configuration composition, not execution.
- **MLflow/W&B:** Our JSONL + analysis charts are sufficient. Adding an experiment tracker would be another dependency for marginal benefit.

### Tools Worth Studying Further

- **doit:** The MD5-based up-to-date check and SQLite state DB are interesting. Could replace our "does file exist?" check with something more robust.
- **joblib.Memory:** The function-level caching pattern could be useful for expensive prep steps (embedding generation, model downloads).
- **AWS Batch Array Jobs:** If we containerize the benchmark runner, array jobs would eliminate the need for SQS + ASG + CDK entirely. One API call to submit up to 10,000 parallel jobs.
