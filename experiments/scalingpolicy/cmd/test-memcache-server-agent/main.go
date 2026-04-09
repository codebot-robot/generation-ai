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
	"bufio"
	"fmt"
	"log"
	"net"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

func getCgroupMemoryLimit() (int64, error) {
	// Try cgroup v2
	data, err := os.ReadFile("/sys/fs/cgroup/memory.max")
	if err == nil {
		val := strings.TrimSpace(string(data))
		if val != "max" {
			v, err := strconv.ParseInt(val, 10, 64)
			if err == nil {
				return v, nil
			}
		}
	}
	// Try cgroup v1
	data, err = os.ReadFile("/sys/fs/cgroup/memory/memory.limit_in_bytes")
	if err == nil {
		val := strings.TrimSpace(string(data))
		v, err := strconv.ParseInt(val, 10, 64)
		if err == nil {
			return v, nil
		}
	}
	return 0, fmt.Errorf("could not find memory limit")
}

func getTargetMemMB() int64 {
	limit, err := getCgroupMemoryLimit()
	if err != nil || limit <= 0 {
		return 64 // fallback
	}
	target := float64(limit) * 0.9
	return int64(target / (1024 * 1024))
}

func updateMemcacheLimit(mb int64) error {
	conn, err := net.DialTimeout("tcp", "127.0.0.1:11211", 5*time.Second)
	if err != nil {
		return err
	}
	defer conn.Close()

	cmd := fmt.Sprintf("cache_memlimit %d\r\n", mb)
	_, err = conn.Write([]byte(cmd))
	if err != nil {
		return err
	}

	reader := bufio.NewReader(conn)
	resp, err := reader.ReadString('\n')
	if err != nil {
		return err
	}
	if !strings.HasPrefix(resp, "OK") {
		return fmt.Errorf("unexpected response: %s", resp)
	}
	return nil
}

func main() {
	initialMB := getTargetMemMB()
	log.Printf("Starting memcached with -m %d", initialMB)

	cmd := exec.Command("memcached", "-m", fmt.Sprintf("%d", initialMB), "-I", "2m")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		log.Fatalf("Failed to start memcached: %v", err)
	}

	currentMB := initialMB

	for {
		time.Sleep(5 * time.Second)
		targetMB := getTargetMemMB()
		if targetMB != currentMB {
			log.Printf("Updating memcache limit to %d MB", targetMB)
			err := updateMemcacheLimit(targetMB)
			if err != nil {
				log.Printf("Failed to update memcache limit: %v", err)
			} else {
				currentMB = targetMB
			}
		}
	}
}
