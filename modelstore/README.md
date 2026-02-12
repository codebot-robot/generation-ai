# Model Store
A simple in-cluster model cache for Hugging Face models.

## Usage
The model store acts as a proxy for Hugging Face Hub. It caches downloaded files locally to avoid repeated downloads from the internet.

### Integration
To use the model store, set the `HF_ENDPOINT` environment variable in your applications to point to the model store service:

```bash
export HF_ENDPOINT=http://modelstore
```

### Deployment
The model store can be deployed to Kubernetes using the provided manifest:

```bash
kubectl apply -f k8s/manifest.yaml
```