# Week 2 Design Report — Demand API Deployment on GKE

**Project:** ops-ai-shreya | **Cluster:** operationalizing-ai (us-central1-a)

---

## Architecture Overview

GitHub pushes trigger CI/CD via GitHub Actions. A Docker image is built and pushed to Artifact Registry, the CD workflow applies Kubernetes manifests and issues a rolling update to a GKE cluster, and the API is exposed via a LoadBalancer at `136.112.78.118`. Pods download model and data files from GCS on startup via an init container.

---

## Deployment Decisions

**Service type — LoadBalancer:** Chosen over NodePort or ClusterIP to provision a stable, directly accessible external IP without additional ingress setup.

**Replicas — 2:** Satisfies the 10+ concurrent requests requirement while keeping costs low on 2-node cluster. Combined with autoscaling (min 2, max 5), handles burst traffic without over-provisioning.

**Rolling update — maxSurge: 1, maxUnavailable: 1:** Ensures at least one replica always serves traffic during deployments while limiting the extra capacity used.

**Resources — 512m/1Gi requests, 1000m/3Gi limits:** Requests sized for LightGBM inference load. Limits set higher (3Gi) because the app expands ~86MB of GCS files into pandas DataFrames at startup, requiring significantly more in-memory space than the raw file size.

**Probe timings — readiness: 60s delay, liveness: 180s delay:** The init container downloads ~86MB from GCS before the main container starts. Readiness at 60s prevents premature traffic routing; liveness was increased to 180s after pods were being killed mid-startup (see below).

**imagePullPolicy — Always:** Ensures pods always pull the latest image tag on restart, avoiding stale cached images during rolling updates.

---

## Debugging & Issues Encountered

**1. ARM64 / AMD64 platform mismatch:** Image built locally on Apple Silicon was rejected by GKE nodes (`ImagePullBackOff`). Rebuilt with `docker buildx build --platform linux/amd64`.

**2. Git LFS breaking the Docker build:** `taxi_zone_lookup.csv` is LFS-tracked. GitHub Actions checked out a pointer file instead of the real CSV, causing `KeyError: 'zone_id'` at startup. Fixed by uploading the CSV to GCS and downloading it via the init container into a shared volume, bypassing the image entirely.

**3. Volume mount path doubling:** Init container wrote CSV to `/metadata-vol/Lookups/taxi_zone_lookup.csv` but the volume was mounted at `/metadata/Lookups` in the main container, making the effective path `/metadata/Lookups/Lookups/...`. Fixed by writing directly to `/metadata-vol/taxi_zone_lookup.csv`.

**4. LightGBM version mismatch:** Forecast endpoint returned `[]` silently. Logs showed `Error loading LightGBM model: unordered_map::at` — model was saved with 4.6.0 but requirements specified 4.1.0. Upgraded `lightgbm==4.1.0 → 4.6.0`.

**5. Liveness probe killing pods mid-startup:** After the LightGBM fix, pods crashed with `CancelledError`. The 90s liveness delay was not enough time to load the 74MB demand parquet into memory. Increased `initialDelaySeconds` from 90 → 180.

**6. Workflows in wrong directory:** GitHub Actions only reads from `.github/workflows/` at the repo root. Workflows were in `week2/starter/.github/workflows/`. Copied to root to activate the pipelines.

**7. CD not applying manifest changes:** The CD workflow only ran `kubectl set image`, ignoring changes to deployment.yaml. Added `kubectl apply -f week2/starter/k8s/` before the image update step.
