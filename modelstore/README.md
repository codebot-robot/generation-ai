# Model Store
A simple in-cluster model cache for Hugging Face models.

## What it is

modelstore keeps model files once, by content, and hands them to
everything in the cluster that needs them:

- **Blobs** are stored by sha256 in a Google Cloud Storage bucket
  (`--blob-store gs://bucket[/prefix]`; a local directory works for
  kind and tests). `PUT /blobs/<sha>` uploads, `GET`/`HEAD /blobs/<sha>`
  reads with HTTP `Range` support, and every blob response carries
  `X-Modelstore-Direct-Url: gs://...` so a reader with its own bucket
  access (the vXPU executor) can range-read the object directly.
- **Imports** pull from the Hugging Face Hub without the bytes passing
  through a client: `POST /blobs/<sha>/import {"url": ...}` fetches one
  file and verifies its digest (`202` while in flight, `201` when
  present, idempotent; the URL must be `https` on the configured Hub
  host, since this pod holds the bucket credentials and is not a
  general fetcher); `POST /models/import {"repo": "org/name",
  "revision": "main"}` imports a whole repository revision in the
  background and records progress on the `Model` object
  (`kubectl get models` shows phase, files and bytes). A download that
  delivers nothing for five minutes is cancelled and the Model marked
  Failed; a restart of the pod resumes any Model it finds in phase
  Importing (blobs already present are skipped). Set `HF_TOKEN` in the
  pod's environment for gated repositories. A repository whose
  normalized name collides with an existing Model from another source
  is refused with `409`.
- **Models** (`kubectl get models`, a cluster-scoped CRD) map file names
  to blobs for one repository revision, with the source repo and the
  commit it resolved to.
- **Hub-compatible reads**: `/api/models/<repo>` and
  `/<repo>/resolve/<rev>/<file>` are served from the Model and its
  blobs, so transformers, vLLM and `huggingface_hub` load from the
  store unchanged with `HF_ENDPOINT=http://modelstore.modelstore`
  (the revision in the path is accepted as-is; the store holds the
  imported one).

## Deploying

One-time, per project: a regional bucket and workload-identity grants.
Readers (vXPU executors and router, test pods in `default`) get
`objectViewer`; the store gets `objectAdmin`.

```sh
PROJECT=justinsb-knotai-dev
NUMBER=$(gcloud projects describe $PROJECT --format='value(projectNumber)')
BUCKET=gs://vxpu-$PROJECT-us-central1
POOL=projects/$NUMBER/locations/global/workloadIdentityPools/$PROJECT.svc.id.goog

gcloud storage buckets create $BUCKET --project=$PROJECT --location=us-central1 \
    --uniform-bucket-level-access
# uploads stage under _tmp/ and are removed after publish; this collects
# any left behind by a pod replaced mid-upload
gcloud storage buckets update $BUCKET --lifecycle-file=<(echo \
    '{"rule":[{"action":{"type":"Delete"},"condition":{"age":1,"matchesPrefix":["_tmp/"]}}]}')
gcloud storage buckets add-iam-policy-binding $BUCKET \
    --member="principal://iam.googleapis.com/$POOL/subject/ns/modelstore/sa/modelstore" \
    --role=roles/storage.objectAdmin
for sa in ns/default/sa/vxpu-executor ns/default/sa/vxpu-router ns/default/sa/default; do
  gcloud storage buckets add-iam-policy-binding $BUCKET \
      --member="principal://iam.googleapis.com/$POOL/subject/$sa" \
      --role=roles/storage.objectViewer
done
```

Then the image and the manifests:

```sh
gcloud builds submit --config cloudbuild.yaml --project=$PROJECT \
    --substitutions _IMAGE=gcr.io/$PROJECT/generation-ai/modelstore:v1 .
kubectl apply -f k8s/crds/
sed -e "s#image: modelstore:latest#image: gcr.io/$PROJECT/generation-ai/modelstore:v1#" \
    -e "s#gs://BUCKET#$BUCKET#" k8s/manifest.yaml | kubectl apply -f -
```

The manifest carries placeholders for the image and the bucket; the
server refuses to start if the bucket is not reachable with the pod's
identity, so a wrong name fails at rollout, not on the first upload.

**Upgrading from the StatefulSet version.** The earlier manifest ran
modelstore as a StatefulSet with a PVC under the same name and Service
selector. `kubectl apply` does not remove it, and the Service would
balance between the old pod and the new one. Delete it first:

```sh
kubectl -n modelstore delete statefulset modelstore --ignore-not-found
kubectl -n modelstore delete pvc -l app=modelstore --ignore-not-found
```

Blobs on the old PVC are not migrated; re-import or re-upload them.

## Usage

```sh
kubectl -n modelstore port-forward svc/modelstore 8080:80 &
# import a repository (returns at once; watch progress)
curl -X POST localhost:8080/models/import -d '{"repo": "Qwen/Qwen3-8B-FP8", "revision": "main"}'
kubectl get models -w
# a file, with a range
curl -r 0-99 localhost:8080/Qwen/Qwen3-8B-FP8/resolve/main/config.json
# a blob by digest, and where it lives
curl -I localhost:8080/blobs/<sha256>
```

From a pod: `HF_ENDPOINT=http://modelstore.modelstore` and the
usual `from_pretrained("Qwen/Qwen3-8B-FP8")`.

### Upload from a directory

`modelstore upload --model-name NAME --model-dir DIR` hashes and uploads
every file, then creates the `Model`.

### Model API (low level)
The model store provides a Kubernetes-native API for managing models.

#### 1. Upload Blobs
Upload model files as blobs identified by their SHA256 hash:
```bash
curl -X PUT --data-binary @weights.bin http://modelstore.modelstore/blobs/<sha256>
```

#### 2. Create Model
Create a `Model` object that references the uploaded blobs:
```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "apiVersion": "generationai.labs.gke.io/v1alpha1",
  "kind": "Model",
  "metadata": { "name": "my-model" },
  "spec": {
    "files": [
      { "path": "weights.bin", "sha256": "<sha256>" }
    ]
  }
}' http://modelstore.modelstore/models
```

#### 3. List Models
```bash
curl http://modelstore.modelstore/models
# or
kubectl get models
```

### Deployment
The model store can be deployed to Kubernetes using the provided manifest:

```bash
# Register the Model CRD
kubectl apply -f k8s/crds/
# Deploy the model store
kubectl apply -f k8s/manifest.yaml
```
