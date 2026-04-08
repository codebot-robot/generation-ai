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

package main

import (
	"context"
	"flag"
	"os"
	"time"

	"github.com/google/cel-go/cel"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	metricsv1beta1 "k8s.io/metrics/pkg/apis/metrics/v1beta1"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"

	scalingpolicyv1alpha1 "github.com/gke-labs/generation-ai/experiments/scalingpolicy/pkg/api/v1alpha1"
)

var (
	scheme   = runtime.NewScheme()
	setupLog = ctrl.Log.WithName("setup")
)

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(scalingpolicyv1alpha1.AddToScheme(scheme))
	utilruntime.Must(metricsv1beta1.AddToScheme(scheme))
}

type ScalingPolicyReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

func (r *ScalingPolicyReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := ctrl.Log.WithName("scalingpolicy-controller")

	var scalingPolicy scalingpolicyv1alpha1.ScalingPolicy
	if err := r.Get(ctx, req.NamespacedName, &scalingPolicy); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	if scalingPolicy.Spec.Target.Kind == "Deployment" {
		var deploy appsv1.Deployment
		if err := r.Get(ctx, types.NamespacedName{
			Name:      scalingPolicy.Spec.Target.Name,
			Namespace: req.Namespace,
		}, &deploy); err != nil {
			return ctrl.Result{}, client.IgnoreNotFound(err)
		}

		// Gather metrics
		envMap := make(map[string]any)
		for _, input := range scalingPolicy.Spec.Inputs {
			if input.Metric == "memory" {
				var podMetricsList metricsv1beta1.PodMetricsList
				err := r.List(ctx, &podMetricsList, client.InNamespace(deploy.Namespace), client.MatchingLabels(deploy.Spec.Selector.MatchLabels))
				if err != nil {
					log.Error(err, "failed to list pod metrics")
					continue
				}
				var maxMem int64
				for _, pm := range podMetricsList.Items {
					for _, container := range pm.Containers {
						mem := container.Usage.Memory().Value()
						if mem > maxMem {
							maxMem = mem
						}
					}
				}
				envMap[input.Name] = maxMem
			}
		}

		updated := false
		for _, val := range scalingPolicy.Spec.Values {
			// Prepare CEL environment with variables
			var celEnvOpts []cel.EnvOption
			for k := range envMap {
				celEnvOpts = append(celEnvOpts, cel.Variable(k, cel.IntType))
			}
			env, err := cel.NewEnv(celEnvOpts...)
			if err != nil {
				log.Error(err, "failed to create CEL env")
				continue
			}
			ast, iss := env.Compile(val.Expression)
			if iss.Err() != nil {
				log.Error(iss.Err(), "failed to compile CEL expression")
				continue
			}
			prg, err := env.Program(ast)
			if err != nil {
				log.Error(err, "failed to create CEL program")
				continue
			}
			out, _, err := prg.Eval(envMap)
			if err != nil {
				log.Error(err, "failed to evaluate CEL expression")
				continue
			}

			var resultInt64 int64
			switch v := out.Value().(type) {
			case int64:
				resultInt64 = int64(v)
			case int:
				resultInt64 = int64(v)
			default:
				log.Error(nil, "CEL expression did not evaluate to integer", "type", out.Type())
				continue
			}

			if val.Min != nil && resultInt64 < int64(*val.Min) {
				resultInt64 = int64(*val.Min)
			}
			if val.Max != nil && resultInt64 > int64(*val.Max) {
				resultInt64 = int64(*val.Max)
			}

			if val.Path == "spec.replicas" {
				if deploy.Spec.Replicas == nil || *deploy.Spec.Replicas != int32(resultInt64) {
					replicas := int32(resultInt64)
					deploy.Spec.Replicas = &replicas
					updated = true
					log.Info("Prepared update Deployment replicas", "replicas", resultInt64)
				}
			} else if val.Path == "spec.template.spec.containers[0].resources.limits.memory" {
				q := resource.NewQuantity(resultInt64, resource.BinarySI)
				if deploy.Spec.Template.Spec.Containers[0].Resources.Limits == nil {
					deploy.Spec.Template.Spec.Containers[0].Resources.Limits = corev1.ResourceList{}
				}
				current := deploy.Spec.Template.Spec.Containers[0].Resources.Limits.Memory()
				if current.Value() != q.Value() {
					deploy.Spec.Template.Spec.Containers[0].Resources.Limits[corev1.ResourceMemory] = *q
					updated = true
					log.Info("Prepared update Deployment memory limit", "memory", q.String())
				}
			}
		}

		if updated {
			if err := r.Update(ctx, &deploy); err != nil {
				log.Error(err, "failed to update deployment")
				return ctrl.Result{}, err
			}
			log.Info("Successfully updated Deployment")
		}
	}

	return ctrl.Result{RequeueAfter: 15 * time.Second}, nil
}
func main() {
	var metricsAddr string
	flag.StringVar(&metricsAddr, "metrics-addr", ":8080", "The address the metric endpoint binds to.")
	flag.Parse()

	ctrl.SetLogger(zap.New(zap.UseDevMode(true)))

	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
		Scheme: scheme,
	})
	if err != nil {
		setupLog.Error(err, "unable to start manager")
		os.Exit(1)
	}

	if err = (&ScalingPolicyReconciler{
		Client: mgr.GetClient(),
		Scheme: mgr.GetScheme(),
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "ScalingPolicy")
		os.Exit(1)
	}

	setupLog.Info("starting manager")
	if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
		setupLog.Error(err, "problem running manager")
		os.Exit(1)
	}
}

func (r *ScalingPolicyReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&scalingpolicyv1alpha1.ScalingPolicy{}).
		Complete(r)
}
