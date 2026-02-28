# Model Store
A simple in-cluster model cache for Hugging Face models.

## Usage
The model store acts as a proxy for Hugging Face Hub. It caches downloaded files locally to avoid repeated downloads from the internet.

### Integration
To use the model store, set the `MODELSTORE_URL` environment variable in your applications to point to the model store service:

```bash
export MODELSTORE_URL=http://modelstore.modelstore
```

### Model API
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
