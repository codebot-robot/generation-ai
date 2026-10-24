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

// #452: per-token cost of reading the n-gram tables' rows, by method.
// A token reads 24 random rows from each of two tables, plus each row's scale: 96 reads.
// usage: rows_bench MODE TOKENS DIR  (MODE: mmap24 | pread24 | pread96 | uring)
// Table layout (from the manifest): per file, weights then scales.
#include <liburing.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

struct Table {
  std::string file;
  long wOff, sOff, rows;
  int fd = -1;
  char* map = nullptr;
  size_t mapLen = 0;
};
static const int kRowsPerTable = 24, kWRow = 256, kSRow = 8;

struct Read { int fd; long off; int len; char* mapped; };

int main(int argc, char** argv) {
  if (argc != 4) { fprintf(stderr, "usage: rows_bench MODE TOKENS DIR\n"); return 2; }
  std::string mode = argv[1], dir = argv[3];
  int tokens = atoi(argv[2]);
  std::vector<Table> tables = {
    {dir + "/824db4881320407ac340736d14dcee5ecd748c27d0f5836b8127ecc2e3781b0f", 664, 98305579672L, 384006168L},
    {dir + "/976330f4954338e1ad8b508c32aa912032c7ad908959fd53c8307650fe4520ed", 672, 98308271264L, 384016682L},
  };
  for (auto& t : tables) {
    t.fd = open(t.file.c_str(), O_RDONLY);
    if (t.fd < 0) { perror(t.file.c_str()); return 1; }
    posix_fadvise(t.fd, 0, 0, POSIX_FADV_RANDOM);  // a miss reads its page, not a readahead window
    if (mode == "mmap24") {
      t.mapLen = lseek(t.fd, 0, SEEK_END);
      t.map = (char*)mmap(nullptr, t.mapLen, PROT_READ, MAP_SHARED, t.fd, 0);
      madvise(t.map, t.mapLen, MADV_RANDOM);
    }
  }
  std::mt19937_64 rng(0);
  std::vector<char> buf(96 * kWRow);
  io_uring ring;
  if (mode == "uring" && io_uring_queue_init(128, &ring, 0) < 0) { perror("io_uring_queue_init"); return 1; }
  std::vector<double> ms;
  volatile long sink = 0;
  // persistent workers (as gather_rows' kernel threads are): each token, worker w takes reads w, w+n, ...
  int nthreads = (mode == "pread96") ? 96 : 24;
  std::atomic<long> gen{0}; std::atomic<int> done{0}; std::atomic<bool> stop{false};
  std::vector<Read>* cur = nullptr;
  std::vector<std::thread> pool;
  if (mode != "uring")
    for (int w = 0; w < nthreads; w++)
      pool.emplace_back([&, w] {
        long seen = 0;
        while (true) {
          long g;
          while ((g = gen.load(std::memory_order_acquire)) == seen && !stop.load()) {}
          if (stop.load()) return;
          seen = g;
          long local = 0;
          auto& rs = *cur;
          for (size_t i = w; i < rs.size(); i += nthreads) {
            if (mode == "mmap24") { for (int b = 0; b < rs[i].len; b += 8) local += rs[i].mapped[b]; }
            else if (pread(rs[i].fd, buf.data() + i * kWRow, rs[i].len, rs[i].off) != rs[i].len) abort();
          }
          sink += local;
          done.fetch_add(1, std::memory_order_acq_rel);
        }
      });
  for (int tok = 0; tok < tokens; tok++) {
    std::vector<Read> reads;
    for (auto& t : tables)
      for (int i = 0; i < kRowsPerTable; i++) {
        long r = rng() % t.rows;
        reads.push_back({t.fd, t.wOff + r * kWRow, kWRow, t.map ? t.map + t.wOff + r * kWRow : nullptr});
        reads.push_back({t.fd, t.sOff + r * kSRow, kSRow, t.map ? t.map + t.sOff + r * kSRow : nullptr});
      }
    auto t0 = std::chrono::steady_clock::now();
    if (mode == "uring") {
      for (size_t i = 0; i < reads.size(); i++) {
        io_uring_sqe* sqe = io_uring_get_sqe(&ring);
        io_uring_prep_read(sqe, reads[i].fd, buf.data() + i * kWRow, reads[i].len, reads[i].off);
      }
      io_uring_submit_and_wait(&ring, reads.size());
      unsigned head, seen = 0; io_uring_cqe* cqe;
      io_uring_for_each_cqe(&ring, head, cqe) { if (cqe->res < 0) { fprintf(stderr, "read: %s\n", strerror(-cqe->res)); return 1; } seen++; }
      io_uring_cq_advance(&ring, seen);
    } else {
      cur = &reads;
      gen.fetch_add(1, std::memory_order_release);  // wake the pool for this token
      while (done.load(std::memory_order_acquire) < nthreads) {}
      done.store(0, std::memory_order_relaxed);
    }
    ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  stop.store(true); for (auto& th : pool) th.join();
  std::vector<double> q = ms; std::sort(q.begin(), q.end());
  double mean = 0; for (double x : ms) mean += x; mean /= ms.size();
  printf("RESULT {\"mode\": \"%s\", \"dir\": \"%s\", \"tokens\": %d, \"reads_per_token\": 96, \"mean_ms\": %.3f, \"p50_ms\": %.3f, \"p99_ms\": %.3f, \"max_ms\": %.3f}\n",
         mode.c_str(), dir.c_str(), tokens, mean, q[q.size() / 2], q[q.size() * 99 / 100], q.back());
  return 0;
}
