package e2e

import (
        "fmt"
        "os"
        "path/filepath"
        "strings"
        "testing"
        "time"
)

func TestE2E(t *testing.T) {
        if os.Getenv("RUN_E2E") == "" {
                t.Skip("Skipping E2E test; RUN_E2E not set")
        }

        h := NewHarness(t, "bema-e2e")
        h.Setup()

        gitRoot := h.GetGitRoot()
        experimentRoot := filepath.Join(gitRoot, "experiments/bema")

        // Build images
        h.DockerBuild("bema:e2e", filepath.Join(experimentRoot, "images/bema/Dockerfile"), experimentRoot)

        // Load images into Kind
        h.KindLoad("bema:e2e")

        // Create namespace and secret first
        h.RunCommand("kubectl", "create", "namespace", "bema")
        h.RunCommand("kubectl", "create", "secret", "generic", "bema", "--from-literal=dummy=value", "-n", "bema")

        // Apply all manifests
        k8sDir := filepath.Join(experimentRoot, "k8s")
        files, err := os.ReadDir(k8sDir)
        if err != nil {
                t.Fatalf("Failed to read k8s dir: %v", err)
        }
        for _, file := range files {
                if file.IsDir() {
                        continue
                }
                path := filepath.Join(k8sDir, file.Name())
                b, err := os.ReadFile(path)
                if err != nil {
                        t.Fatalf("Failed to read manifest %s: %v", path, err)
                }
                manifest := string(b)
                if file.Name() == "manifest.yaml" {
                        manifest = strings.ReplaceAll(manifest, "image: bema:latest", "image: bema:e2e\n          imagePullPolicy: Never")
                }
                if file.Name() == "cert-manager.yaml" {
                        h.RunCommand("kubectl", "apply", "-f", "https://github.com/cert-manager/cert-manager/releases/download/v1.17.1/cert-manager.yaml")
                        time.Sleep(30 * time.Second)
                }
                h.KubectlApplyContent(file.Name(), manifest)
        }

        // Wait for server
        if err := h.WaitForStatefulSet("bema", "bema", 2*time.Minute); err != nil {
                fmt.Fprintf(os.Stderr, "Bema failed to start: %v\n", err)
                fmt.Fprintf(os.Stderr, "Bema Pod YAML:\n%s\n", h.GetPodYaml("app=bema", "bema"))
                fmt.Fprintf(os.Stderr, "Events:\n%s\n", h.GetEvents("bema"))
                t.Fatalf("Bema failed to start: %v", err)
        }
}
