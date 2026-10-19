// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package router

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
	"github.com/gke-labs/generation-ai/vxpu/pkg/blobserver"
	"github.com/gke-labs/generation-ai/vxpu/pkg/weightstore"
)

const testCommit = "0123456789abcdef0123"

var testManifest = `{"format":"vxpu-manifest/v1alpha1","source":{"repo":"org/M","revision":"main","commit_sha":"` + testCommit + `"},` +
	`"config":{},"files":{"abc":{"size":42,"name":"model.safetensors","source":"https://huggingface.co/org/M/resolve/c/model.safetensors"}},` +
	`"tensors":{}}`

// fakeModelstore serves the routes the catalog uses for one model, org/M.
type fakeModelstore struct {
	mu       sync.Mutex
	compiled bool // the artifact for testCommit exists
}

func (f *fakeModelstore) handler(t *testing.T) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.mu.Lock()
		compiled := f.compiled
		f.mu.Unlock()
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/models/import":
			w.WriteHeader(http.StatusAccepted)
			_, _ = w.Write([]byte(`{"metadata":{"name":"org/m"}}`))
		case r.URL.Path == "/models/org/m":
			_, _ = w.Write([]byte(`{"status":{"phase":"Ready"}}`))
		case r.URL.Path == "/v2/org/m/manifests/main":
			_, _ = w.Write([]byte(`{"annotations":{"vnd.gke-labs.huggingface.commit":"` + testCommit + `"}}`))
		case strings.HasPrefix(r.URL.Path, "/v2/vxpu-artifacts/org/m/manifests/"):
			if !compiled {
				http.NotFound(w, r)
			}
		case strings.HasPrefix(r.URL.Path, "/vxpu-artifacts/org/m/resolve/") && compiled:
			switch r.URL.Path[strings.LastIndex(r.URL.Path, "/")+1:] {
			case "manifest.json":
				_, _ = w.Write([]byte(testManifest))
			case "binding.json":
				_, _ = w.Write([]byte(`{}`))
			default:
				_, _ = w.Write([]byte("graph"))
			}
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/import"):
			w.WriteHeader(http.StatusCreated)
		default:
			t.Logf("fake modelstore: unexpected %s %s", r.Method, r.URL.Path)
			http.NotFound(w, r)
		}
	})
}

// loadingExecutor is a fake executor that records loads and can be made
// to forget its model (as an idle keep-alive eviction does).
type loadingExecutor struct {
	pb.UnimplementedExecutorServer
	mu           sync.Mutex
	loads        int
	evicted      bool
	loadFailed   bool   // its asynchronous load failed: NewSession says so until the next LoadModel
	lastManifest string // the manifest of the last LoadModel
}

func (e *loadingExecutor) LoadModel(_ context.Context, req *pb.LoadModelRequest) (*pb.LoadModelResponse, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if string(req.DecodeGraph) != "graph" || !strings.Contains(req.ManifestJson, "/blobs/abc") {
		return nil, status.Error(codes.InvalidArgument, "unexpected artifact")
	}
	e.loads++
	e.evicted = false
	e.loadFailed = false
	e.lastManifest = req.ManifestJson
	return &pb.LoadModelResponse{Success: true}, nil
}

func (e *loadingExecutor) NewSession(context.Context, *pb.NewSessionRequest) (*pb.NewSessionResponse, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.evicted {
		return nil, status.Error(codes.FailedPrecondition, "call LoadModel first")
	}
	if e.loadFailed {
		return nil, status.Error(codes.Internal, "model load failed: weights unreachable")
	}
	return &pb.NewSessionResponse{SessionId: "s1"}, nil
}

func newCatalogServer(t *testing.T, store *fakeModelstore) (*Server, *fake.Clientset) {
	t.Helper()
	ms := httptest.NewServer(store.handler(t))
	t.Cleanup(ms.Close)
	bs, err := blobserver.NewServer(t.TempDir(), 0)
	if err != nil {
		t.Fatal(err)
	}
	clientset := fake.NewClientset()
	s := &Server{
		clientset: clientset, namespace: "ns", imageName: "registry/vxpu-executor:v42", blobServer: bs,
		weightStore: weightstore.New(ms.URL), executorPort: 50051,
		peerToModel: map[string]string{}, externalModel: map[string]string{}, modelEndpoint: map[string]string{},
		loadedTags: map[string]string{}, compileResources: defaultCompileResources(), compilePoll: 10 * time.Millisecond,
	}
	return s, clientset
}

func TestArtifactTag(t *testing.T) {
	for image, want := range map[string]string{
		"gcr.io/p/vxpu-executor:v42":           "0123456789ab_vxpu-executor-v42",
		"localhost:5000/vxpu-executor":         "0123456789ab_vxpu-executor-latest",
		"vxpu-executor:e2e":                    "0123456789ab_vxpu-executor-e2e",
		"registry-b/other-executor:latest":     "0123456789ab_other-executor-latest", // not the same entry as vxpu-executor:latest
		"gcr.io/p/x@sha256:" + testCommit + "": "0123456789ab_0123456789ab",
	} {
		if got := artifactTag(testCommit, image); got != want {
			t.Errorf("artifactTag(%q) = %q, want %q", image, got, want)
		}
	}
}

func TestRegisterModel_AlreadyCompiled(t *testing.T) {
	s, clientset := newCatalogServer(t, &fakeModelstore{compiled: true})
	resp, err := s.RegisterModel(t.Context(), &pb.RegisterModelRequest{Model: "org/M"})
	if err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}
	if resp.Compiled || resp.Commit != testCommit || resp.Artifact != "vxpu-artifacts/org/m:0123456789ab_vxpu-executor-v42" || resp.WeightBytes != 42 {
		t.Errorf("unexpected response %+v", resp)
	}
	if jobs, _ := clientset.BatchV1().Jobs("ns").List(t.Context(), metav1.ListOptions{}); len(jobs.Items) != 0 {
		t.Errorf("no compile job expected, got %d", len(jobs.Items))
	}
}

func TestRegisterModel_RunsCompileJob(t *testing.T) {
	store := &fakeModelstore{}
	s, clientset := newCatalogServer(t, store)
	// The "cluster": when the compile job appears, it succeeds and the
	// artifact appears in modelstore.
	ctx := t.Context()
	go func() {
		for {
			jobs, err := clientset.BatchV1().Jobs("ns").List(ctx, metav1.ListOptions{})
			if err == nil && len(jobs.Items) == 1 {
				job := jobs.Items[0]
				store.mu.Lock()
				store.compiled = true
				store.mu.Unlock()
				job.Status.Succeeded = 1
				_, _ = clientset.BatchV1().Jobs("ns").UpdateStatus(ctx, &job, metav1.UpdateOptions{})
				return
			}
			time.Sleep(5 * time.Millisecond)
		}
	}()
	resp, err := s.RegisterModel(ctx, &pb.RegisterModelRequest{Model: "org/M"})
	if err != nil {
		t.Fatalf("RegisterModel: %v", err)
	}
	if !resp.Compiled || resp.WeightBytes != 42 {
		t.Errorf("unexpected response %+v", resp)
	}
	jobs, _ := clientset.BatchV1().Jobs("ns").List(t.Context(), metav1.ListOptions{})
	if len(jobs.Items) != 1 {
		t.Fatalf("one compile job expected, got %d", len(jobs.Items))
	}
	c := jobs.Items[0].Spec.Template.Spec.Containers[0]
	args := strings.Join(append(append([]string{}, c.Command...), c.Args...), " ")
	for _, want := range []string{"python -m vxpu.compile org/M", "--tag 0123456789ab_vxpu-executor-v42", "--revision " + testCommit, "--modelstore http"} {
		if !strings.Contains(args, want) {
			t.Errorf("compile job command %q lacks %q", args, want)
		}
	}
	if c.Image != "registry/vxpu-executor:v42" {
		t.Errorf("compile job image %q, want the executor image", c.Image)
	}
}

func TestNewSessionByName_ExternalExecutor(t *testing.T) {
	ctx := t.Context()
	s, _ := newCatalogServer(t, &fakeModelstore{compiled: true})
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	executor := &loadingExecutor{}
	srv := grpc.NewServer()
	pb.RegisterExecutorServer(srv, executor)
	go func() { _ = srv.Serve(lis) }()
	defer srv.Stop()
	s.UseExternalExecutors([]string{lis.Addr().String()})

	// unregistered models are refused
	if _, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/Other"}); status.Code(err) != codes.FailedPrecondition {
		t.Errorf("unregistered model: got %v, want FAILED_PRECONDITION", err)
	}

	resp, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"})
	if err != nil {
		t.Fatalf("NewSession by name: %v", err)
	}
	modelKey, backend, err := splitSessionID(resp.SessionId)
	if err != nil || backend != "s1" {
		t.Fatalf("session id %q: %v", resp.SessionId, err)
	}
	if _, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"}); err != nil {
		t.Fatalf("second NewSession: %v", err)
	}
	if executor.loads != 1 {
		t.Errorf("the artifact should be loaded once, loaded %d times", executor.loads)
	}
	// sessions route to the external executor, not to a pod
	if endpoint, err := s.executorEndpoint(ctx, modelKey); err != nil || endpoint != lis.Addr().String() {
		t.Errorf("executorEndpoint = %q, %v; want the external executor", endpoint, err)
	}
	// the executor evicts the model: the next call says it is being
	// reloaded, and the one after that loads it again
	executor.mu.Lock()
	executor.evicted = true
	executor.mu.Unlock()
	if _, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"}); status.Code(err) != codes.FailedPrecondition {
		t.Errorf("after eviction: got %v, want FAILED_PRECONDITION", err)
	}
	if _, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"}); err != nil {
		t.Fatalf("NewSession after eviction: %v", err)
	}
	if executor.loads != 2 {
		t.Errorf("the evicted model should be loaded again, loads = %d", executor.loads)
	}
}

// startExecutor serves a fake executor on a local port.
func startExecutor(t *testing.T, executor pb.ExecutorServer) string {
	t.Helper()
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	srv := grpc.NewServer()
	pb.RegisterExecutorServer(srv, executor)
	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)
	return lis.Addr().String()
}

func TestNewSessionByName_FailedLoadIsForgotten(t *testing.T) {
	ctx := t.Context()
	s, _ := newCatalogServer(t, &fakeModelstore{compiled: true})
	executor := &loadingExecutor{}
	s.UseExternalExecutors([]string{startExecutor(t, executor)})
	if _, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"}); err != nil {
		t.Fatalf("NewSession by name: %v", err)
	}
	// the executor's asynchronous load fails (it answers so until the next
	// LoadModel): the router forgets the placement instead of passing the
	// same error on forever, and the next request loads the model again
	executor.mu.Lock()
	executor.loadFailed = true
	executor.mu.Unlock()
	if _, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"}); status.Code(err) != codes.Unavailable {
		t.Errorf("after a failed load: got %v, want UNAVAILABLE", err)
	}
	if _, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"}); err != nil {
		t.Fatalf("NewSession after a failed load: %v", err)
	}
	if executor.loads != 2 {
		t.Errorf("the model should be loaded again after its load failed, loads = %d", executor.loads)
	}
}

func TestNewSessionByName_ExternalModelstoreURL(t *testing.T) {
	ctx := t.Context()
	s, _ := newCatalogServer(t, &fakeModelstore{compiled: true})
	executor := &loadingExecutor{}
	s.UseExternalExecutors([]string{startExecutor(t, executor)})
	s.UseExternalModelstoreURL("http://10.0.0.7:8080/")
	if _, err := s.NewSession(ctx, &pb.NewSessionRequest{Model: "org/M"}); err != nil {
		t.Fatalf("NewSession by name: %v", err)
	}
	if !strings.Contains(executor.lastManifest, `"http://10.0.0.7:8080/blobs/abc"`) {
		t.Errorf("an external executor should fetch from the external modelstore URL; its manifest: %s", executor.lastManifest)
	}
}

func TestRegisterModel_ReplacesFinishedCompileJob(t *testing.T) {
	store := &fakeModelstore{}
	s, clientset := newCatalogServer(t, store)
	ctx := t.Context()
	// a compile that failed earlier, its Job still present (TTL 24 h)
	name := compileJobName(weightstore.ArtifactRepository("org/M"), artifactTag(testCommit, s.imageName))
	stale := &batchv1.Job{ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "ns"},
		Status: batchv1.JobStatus{Conditions: []batchv1.JobCondition{{Type: batchv1.JobFailed, Status: corev1.ConditionTrue}}}}
	if _, err := clientset.BatchV1().Jobs("ns").Create(ctx, stale, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
	// the "cluster": a fresh compile job (no conditions) succeeds and the
	// artifact appears in modelstore
	go func() {
		for {
			job, err := clientset.BatchV1().Jobs("ns").Get(ctx, name, metav1.GetOptions{})
			if err == nil && len(job.Status.Conditions) == 0 && job.Status.Succeeded == 0 {
				store.mu.Lock()
				store.compiled = true
				store.mu.Unlock()
				job.Status.Succeeded = 1
				_, _ = clientset.BatchV1().Jobs("ns").UpdateStatus(ctx, job, metav1.UpdateOptions{})
				return
			}
			time.Sleep(5 * time.Millisecond)
		}
	}()
	resp, err := s.RegisterModel(ctx, &pb.RegisterModelRequest{Model: "org/M"})
	if err != nil {
		t.Fatalf("RegisterModel with a stale failed compile job: %v", err)
	}
	if !resp.Compiled {
		t.Errorf("the model should have been compiled again, got %+v", resp)
	}
}
