#!/usr/bin/env bash
#
# One-time GCP infrastructure provisioning.
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - A GCP project already created
#
# Usage:
#   export GCP_PROJECT_ID=your-project-id
#   export GCP_REGION=us-central1
#   export GITHUB_ORG=your-github-username
#   export GITHUB_REPO=your-repo-name
#   bash infra/gcp/setup.sh
#
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"
REGION="${GCP_REGION:-us-central1}"
GITHUB_ORG="${GITHUB_ORG:?Set GITHUB_ORG}"
GITHUB_REPO="${GITHUB_REPO:?Set GITHUB_REPO}"

SA_NAME="github-actions"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
WIF_POOL="github-pool"
WIF_PROVIDER="github-provider"
AR_REPO="market-predictor"

echo "=== Setting project to ${PROJECT_ID} ==="
gcloud config set project "${PROJECT_ID}"

# ── Enable APIs ──────────────────────────────────────────────
echo "=== Enabling APIs ==="
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  secretmanager.googleapis.com \
  aiplatform.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com \
  iamcredentials.googleapis.com \
  iam.googleapis.com \
  cloudresourcemanager.googleapis.com

# ── Artifact Registry ────────────────────────────────────────
echo "=== Creating Artifact Registry repo ==="
gcloud artifacts repositories create "${AR_REPO}" \
  --repository-format=docker \
  --location="${REGION}" \
  --description="Market Predictor Docker images" \
  2>/dev/null || echo "  (already exists)"

# ── GCS Buckets ──────────────────────────────────────────────
echo "=== Creating GCS buckets ==="
gsutil mb -p "${PROJECT_ID}" -l "${REGION}" "gs://${PROJECT_ID}-ml-artifacts" 2>/dev/null || echo "  ml-artifacts bucket already exists"
gsutil mb -p "${PROJECT_ID}" -l "${REGION}" "gs://${PROJECT_ID}-dvc-store" 2>/dev/null || echo "  dvc-store bucket already exists"

# ── Secret Manager ───────────────────────────────────────────
echo "=== Creating secrets (add values manually later) ==="
for SECRET in TWELVE_DATA_API_KEY ALPHA_VANTAGE_API_KEY; do
  gcloud secrets create "${SECRET}" --replication-policy=automatic 2>/dev/null || echo "  ${SECRET} already exists"
done

echo ""
echo "  To set secret values:"
echo "    echo -n 'YOUR_KEY' | gcloud secrets versions add TWELVE_DATA_API_KEY --data-file=-"
echo "    echo -n 'YOUR_KEY' | gcloud secrets versions add ALPHA_VANTAGE_API_KEY --data-file=-"
echo ""

# ── Service Account ──────────────────────────────────────────
echo "=== Creating service account ==="
gcloud iam service-accounts create "${SA_NAME}" \
  --display-name="GitHub Actions CI/CD" \
  2>/dev/null || echo "  (already exists)"

ROLES=(
  "roles/run.developer"
  "roles/artifactregistry.writer"
  "roles/storage.objectAdmin"
  "roles/secretmanager.secretAccessor"
  "roles/aiplatform.user"
  "roles/iam.serviceAccountUser"
)

for ROLE in "${ROLES[@]}"; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="${ROLE}" \
    --condition=None \
    --quiet
done

# ── Workload Identity Federation ─────────────────────────────
echo "=== Setting up Workload Identity Federation ==="
gcloud iam workload-identity-pools create "${WIF_POOL}" \
  --location="global" \
  --display-name="GitHub Actions Pool" \
  2>/dev/null || echo "  Pool already exists"

gcloud iam workload-identity-pools providers create-oidc "${WIF_PROVIDER}" \
  --location="global" \
  --workload-identity-pool="${WIF_POOL}" \
  --display-name="GitHub OIDC Provider" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  2>/dev/null || echo "  Provider already exists"

PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')

gcloud iam service-accounts add-iam-policy-binding "${SA_EMAIL}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${WIF_POOL}/attribute.repository/${GITHUB_ORG}/${GITHUB_REPO}" \
  --quiet

# ── Cloud Run Placeholder Services ───────────────────────────
echo "=== Creating placeholder Cloud Run services ==="
for SVC in market-predictor-test market-predictor-prod; do
  gcloud run services describe "${SVC}" --region="${REGION}" --quiet 2>/dev/null || \
    echo "  ${SVC} will be created on first deploy"
done

# ── Print summary ────────────────────────────────────────────
echo ""
echo "============================================"
echo " GCP Setup Complete"
echo "============================================"
echo ""
echo " GitHub Secrets to configure:"
echo "   GCP_PROJECT_ID    = ${PROJECT_ID}"
echo "   GCP_SA_EMAIL      = ${SA_EMAIL}"
echo "   WIF_PROVIDER      = projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${WIF_POOL}/providers/${WIF_PROVIDER}"
echo ""
echo " Artifact Registry:"
echo "   ${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/app"
echo ""
echo " GCS Buckets:"
echo "   gs://${PROJECT_ID}-ml-artifacts"
echo "   gs://${PROJECT_ID}-dvc-store"
echo ""
