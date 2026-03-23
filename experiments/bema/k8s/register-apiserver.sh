#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

# Configuration
NAMESPACE="${BEMA_NAMESPACE:-bema}"
SERVICE_NAME="bema"
APISERVER_PORT="8080"
SECRET_NAME="bema-apiserver-tls"
GROUP="bema.labs.gke.io"
VERSION="v1alpha1"

echo "Registering Bema aggregated API server..."

# Generate certificates
mkdir -p .certs
openssl req -x509 -nodes -days 365 -newkey rsa:2048 
  -keyout .certs/server.key -out .certs/server.crt 
  -subj "/CN=${SERVICE_NAME}.${NAMESPACE}.svc" 
  -addext "subjectAltName = DNS:${SERVICE_NAME}.${NAMESPACE}.svc"

# Create TLS secret
kubectl create secret tls ${SECRET_NAME} 
  --cert=.certs/server.crt 
  --key=.certs/server.key 
  -n ${NAMESPACE} 
  --dry-run=client -o yaml | kubectl apply -f -

# Patch StatefulSet to use TLS
# We use kubectl patch to add the TLS flags and volumes to the existing StatefulSet.
kubectl patch statefulset bema -n ${NAMESPACE} --type='json' -p="[
  {"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--tls-cert-file=/etc/tls/tls.crt"},
  {"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--tls-key-file=/etc/tls/tls.key"},
  {"op": "add", "path": "/spec/template/spec/containers/0/volumeMounts/-", "value": {"name": "tls", "mountPath": "/etc/tls", "readOnly": true}},
  {"op": "add", "path": "/spec/template/spec/volumes/-", "value": {"name": "tls", "secret": {"secretName": "${SECRET_NAME}"}}}
]"

# Register APIService
CA_BUNDLE=$(cat .certs/server.crt | base64 | tr -d '
')

cat <<EOF | kubectl apply -f -
apiVersion: apiregistration.k8s.io/v1
kind: APIService
metadata:
  name: ${VERSION}.${GROUP}
spec:
  service:
    name: ${SERVICE_NAME}
    namespace: ${NAMESPACE}
    port: ${APISERVER_PORT}
  group: ${GROUP}
  version: ${VERSION}
  groupPriorityMinimum: 100
  versionPriority: 100
  caBundle: ${CA_BUNDLE}
EOF

echo "Bema aggregated API server registered!"
echo "You can now use kubectl to interact with Bema resources, e.g.:"
echo "  kubectl get chatsessions"
