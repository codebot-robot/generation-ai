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
	"fmt"
	"os"
	"strings"

	"github.com/gke-labs/generation-ai/vxpu/pkg/kernelplan"
	"k8s.io/klog/v2"
)

// vxpu kernels plan --record R [--record R2] [--out table.json]
// vxpu kernels check --table T --record R
//
// The kernel planner (pkg/kernelplan): from the records engines wrote
// (VXPU_PLAN_RECORD), the table an engine applies (VXPU_PLAN_TABLE); and
// the check that a run applied its table.
func cmdKernels(ctx context.Context, args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("usage: vxpu kernels plan|check ...")
	}
	switch args[0] {
	case "plan":
		return runKernelsPlan(ctx, args[1:])
	case "check":
		return runKernelsCheck(ctx, args[1:])
	}
	return fmt.Errorf("vxpu kernels: unknown subcommand %q (plan, check)", args[0])
}

type stringList []string

func (l *stringList) String() string     { return strings.Join(*l, ",") }
func (l *stringList) Set(s string) error { *l = append(*l, s); return nil }

func runKernelsPlan(ctx context.Context, args []string) error {
	log := klog.FromContext(ctx)
	flags := flag.NewFlagSet("kernels plan", flag.ContinueOnError)
	var paths stringList
	flags.Var(&paths, "record", "a kernel record an engine wrote (repeatable; one GPU)")
	out := flags.String("out", "", "where to write the table (default: stdout)")
	launch := flags.Float64("launch-us", 0, "one kernel launch in a CUDA graph, credited to a fused body (default: the record's launch_us)")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if len(paths) == 0 {
		return fmt.Errorf("vxpu kernels plan: at least one --record")
	}
	var records []*kernelplan.Record
	for _, p := range paths {
		r, err := kernelplan.ReadRecordFile(p)
		if err != nil {
			return err
		}
		log.Info("read a kernel record", "path", p, "gpu", r.Hardware.GPU, "sites", len(r.Sites))
		records = append(records, r)
	}
	table, err := kernelplan.Plan(ctx, records, kernelplan.Options{LaunchUs: *launch})
	if err != nil {
		return err
	}
	data, err := table.Write()
	if err != nil {
		return err
	}
	for _, line := range table.Explain {
		fmt.Fprintln(os.Stderr, line)
	}
	if *out == "" {
		fmt.Println(string(data))
		return nil
	}
	return os.WriteFile(*out, data, 0o644)
}

func runKernelsCheck(ctx context.Context, args []string) error {
	flags := flag.NewFlagSet("kernels check", flag.ContinueOnError)
	tablePath := flags.String("table", "", "the kernel table the engine was given")
	recordPath := flags.String("record", "", "the kernel record the engine wrote")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *tablePath == "" || *recordPath == "" {
		return fmt.Errorf("vxpu kernels check: --table and --record")
	}
	table, err := kernelplan.ReadTableFile(*tablePath)
	if err != nil {
		return err
	}
	record, err := kernelplan.ReadRecordFile(*recordPath)
	if err != nil {
		return err
	}
	bad := kernelplan.Check(table, record)
	fmt.Printf("%d sites, %d did not run the table's body\n", len(record.Sites), len(bad))
	for _, m := range bad {
		fmt.Println("  " + m.String())
	}
	if len(bad) > 0 {
		return fmt.Errorf("the table was not applied at %d sites", len(bad))
	}
	return nil
}
