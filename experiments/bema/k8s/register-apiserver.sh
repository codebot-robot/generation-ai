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
APISERVER_PORT="443"
SECRET_NAME="bema-apiserver-tls"
GROUP="bema.labs.gke.io"
VERSION="v1alpha1"

echo "Registering Bema aggregated API server..."

# Generate certificates
mkdir -p .certs
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout .certs/server.key -out .certs/server.crt \
  -subj "/CN=${SERVICE_NAME}.${NAMESPACE}.svc" \
  -addext "subjectAltName = DNS:${SERVICE_NAME}.${NAMESPACE}.svc"

# Create TLS secret
kubectl create secret tls ${SECRET_NAME} \
  --cert=.certs/server.crt \
  --key=.certs/server.key \
  -n ${NAMESPACE} \
  --dry-run=client -o yaml | kubectl apply -f -

# Patch StatefulSet to use TLS
# We use a python script for more robust patching of lists that might not exist.
python3 - <<EOF
import json, subprocess, sys

NAMESPACE="${NAMESPACE}"
SECRET_NAME="${SECRET_NAME}"

def patch():
    try:
        ss = json.loads(subprocess.check_output(["kubectl", "get", "statefulset", "bema", "-n", NAMESPACE, "-o", "json"]))
    except subprocess.CalledProcessError:
        print("StatefulSet bema not found in namespace " + NAMESPACE)
        sys.exit(1)

    # In StatefulSet, the pod spec is in .spec.template.spec
    pod_spec = ss["spec"]["template"]["spec"]
    container = pod_spec["containers"][0]

    # Add TLS args
    args = container.get("args", [])
    tls_args = ["--tls-cert-file=/etc/tls/tls.crt", "--tls-key-file=/etc/tls/tls.key"]
    for arg in tls_args:
        if arg not in args:
            args.append(arg)
    container["args"] = args

    # Add volume mount
    vms = container.get("volumeMounts", [])
    if not any(vm["name"] == "tls" for vm in vms):
        vms.append({"name": "tls", "mountPath": "/etc/tls", "readOnly": True})
    container["volumeMounts"] = vms

    # Add volume
    volumes = pod_spec.get("volumes", [])
    if not any(v["name"] == "tls" for v in volumes):
        volumes.append({"name": "tls", "secret": {"secretName": SECRET_NAME}})
    pod_spec["volumes"] = volumes

    # We use a strategic merge patch style update
    patch_data = {
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": container["name"],
                            "args": args,
                            "volumeMounts": vms
                        }
                    ],
                    "volumes": volumes
                }
            }
        }
    }
    
    subprocess.run(["kubectl", "patch", "statefulset", "bema", "-n", NAMESPACE, "--patch", json.dumps(patch_data)], check=True)

patch()
EOF

# Register APIService
CA_BUNDLE=$(cat .certs/server.crt | base64 | tr -d '\n')

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
