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

package e2e

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	commonpb "go.opentelemetry.io/proto/otlp/common/v1"
	metricspb "go.opentelemetry.io/proto/otlp/metrics/v1"
	resourcepb "go.opentelemetry.io/proto/otlp/resource/v1"
	"google.golang.org/protobuf/encoding/protojson"
	"k8s.io/apimachinery/pkg/api/resource"
)

type MetricsSink struct {
	file *os.File
}

func NewMetricsSink(t *testing.T) *MetricsSink {
	artifactsDir := os.Getenv("ARTIFACTS")
	if artifactsDir == "" {
		artifactsDir = "/tmp"
	}

	testArtifactsDir := filepath.Join(artifactsDir, t.Name())
	if err := os.MkdirAll(testArtifactsDir, 0755); err != nil {
		t.Fatalf("Failed to create test artifacts directory: %v", err)
	}

	jsonlPath := filepath.Join(testArtifactsDir, "metrics.jsonl")
	f, err := os.OpenFile(jsonlPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		t.Fatalf("Failed to open jsonl file: %v", err)
	}

	t.Cleanup(func() {
		f.Close()
	})

	return &MetricsSink{
		file: f,
	}
}

func (s *MetricsSink) RecordMetrics(batch *metricspb.ResourceMetrics) error {
	b, err := protojson.Marshal(batch)
	if err != nil {
		return err
	}
	_, err = s.file.Write(append(b, '\n'))
	return err
}

func (s *MetricsSink) StartCollecting(t *testing.T, namespace string) {
	ctx, cancel := context.WithCancel(t.Context())
	t.Cleanup(cancel)

	go func() {
		ticker := time.NewTicker(1 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				s.collectOnce(t, namespace)
			}
		}
	}()
}

func createMetric(name, pod, container string, val float64, now uint64) *metricspb.Metric {
	return &metricspb.Metric{
		Name: name,
		Data: &metricspb.Metric_Gauge{
			Gauge: &metricspb.Gauge{
				DataPoints: []*metricspb.NumberDataPoint{
					{
						TimeUnixNano: now,
						Value:        &metricspb.NumberDataPoint_AsDouble{AsDouble: val},
						Attributes: []*commonpb.KeyValue{
							{
								Key:   "pod",
								Value: &commonpb.AnyValue{Value: &commonpb.AnyValue_StringValue{StringValue: pod}},
							},
							{
								Key:   "container",
								Value: &commonpb.AnyValue{Value: &commonpb.AnyValue_StringValue{StringValue: container}},
							},
						},
					},
				},
			},
		},
	}
}

func parseK8sQuantity(s string) float64 {
	q, err := resource.ParseQuantity(s)
	if err != nil {
		return 0
	}
	return q.AsApproximateFloat64()
}

func (s *MetricsSink) collectOnce(t *testing.T, namespace string) {
	now := uint64(time.Now().UnixNano())
	rm := &metricspb.ResourceMetrics{
		Resource: &resourcepb.Resource{
			Attributes: []*commonpb.KeyValue{
				{
					Key:   "namespace",
					Value: &commonpb.AnyValue{Value: &commonpb.AnyValue_StringValue{StringValue: namespace}},
				},
			},
		},
		ScopeMetrics: []*metricspb.ScopeMetrics{
			{
				Scope: &commonpb.InstrumentationScope{
					Name: "e2e-test",
				},
			},
		},
	}

	metrics := []*metricspb.Metric{}

	// 1. App Metrics
	cmd := exec.Command("kubectl", "exec", "deployment/memcache-client", "-n", namespace, "--", "curl", "-s", "http://memcache-service:8080/metrics")
	out, err := cmd.Output()
	if err == nil {
		for _, line := range strings.Split(string(out), "\n") {
			if strings.HasPrefix(line, "#") || strings.TrimSpace(line) == "" {
				continue
			}
			parts := strings.Fields(line)
			if len(parts) == 2 {
				name := parts[0]
				val, err := strconv.ParseFloat(parts[1], 64)
				if err == nil {
					m := &metricspb.Metric{
						Name: name,
						Data: &metricspb.Metric_Gauge{
							Gauge: &metricspb.Gauge{
								DataPoints: []*metricspb.NumberDataPoint{
									{
										TimeUnixNano: now,
										Value:        &metricspb.NumberDataPoint_AsDouble{AsDouble: val},
									},
								},
							},
						},
					}
					metrics = append(metrics, m)
				}
			}
		}
	}

	// 2. Pod Metrics
	cmd = exec.Command("kubectl", "get", "--raw", fmt.Sprintf("/apis/metrics.k8s.io/v1beta1/namespaces/%s/pods", namespace))
	out, err = cmd.Output()
	if err == nil {
		var podMetrics struct {
			Items []struct {
				Metadata struct {
					Name string `json:"name"`
				} `json:"metadata"`
				Containers []struct {
					Name  string            `json:"name"`
					Usage map[string]string `json:"usage"`
				} `json:"containers"`
			} `json:"items"`
		}
		if json.Unmarshal(out, &podMetrics) == nil {
			for _, pod := range podMetrics.Items {
				for _, container := range pod.Containers {
					if cpuStr, ok := container.Usage["cpu"]; ok {
						metrics = append(metrics, createMetric("pod_cpu_usage", pod.Metadata.Name, container.Name, parseK8sQuantity(cpuStr), now))
					}
					if memStr, ok := container.Usage["memory"]; ok {
						metrics = append(metrics, createMetric("pod_memory_usage", pod.Metadata.Name, container.Name, parseK8sQuantity(memStr), now))
					}
				}
			}
		}
	}

	// 3. Pod requests and limits
	cmd = exec.Command("kubectl", "get", "pods", "-n", namespace, "-o", "json")
	out, err = cmd.Output()
	if err == nil {
		var pods struct {
			Items []struct {
				Metadata struct {
					Name string `json:"name"`
				} `json:"metadata"`
				Spec struct {
					Containers []struct {
						Name      string `json:"name"`
						Resources struct {
							Requests map[string]string `json:"requests"`
							Limits   map[string]string `json:"limits"`
						} `json:"resources"`
					} `json:"containers"`
				} `json:"spec"`
			} `json:"items"`
		}
		if json.Unmarshal(out, &pods) == nil {
			for _, pod := range pods.Items {
				for _, container := range pod.Spec.Containers {
					if cpuStr, ok := container.Resources.Requests["cpu"]; ok {
						metrics = append(metrics, createMetric("pod_cpu_request", pod.Metadata.Name, container.Name, parseK8sQuantity(cpuStr), now))
					}
					if memStr, ok := container.Resources.Requests["memory"]; ok {
						metrics = append(metrics, createMetric("pod_memory_request", pod.Metadata.Name, container.Name, parseK8sQuantity(memStr), now))
					}
					if cpuStr, ok := container.Resources.Limits["cpu"]; ok {
						metrics = append(metrics, createMetric("pod_cpu_limit", pod.Metadata.Name, container.Name, parseK8sQuantity(cpuStr), now))
					}
					if memStr, ok := container.Resources.Limits["memory"]; ok {
						metrics = append(metrics, createMetric("pod_memory_limit", pod.Metadata.Name, container.Name, parseK8sQuantity(memStr), now))
					}
				}
			}
		}
	}

	if len(metrics) > 0 {
		rm.ScopeMetrics[0].Metrics = metrics
		s.RecordMetrics(rm)
	}
}
