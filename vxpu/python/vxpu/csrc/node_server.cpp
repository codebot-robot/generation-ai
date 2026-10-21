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

// The native data path of a decode step's host nodes: no Python between
// messages.
//
//   NodeServer  (#394) dataflow nodes, as AOTInductor packages, answering
//               their mailboxes over TCP. Python builds the packages and
//               loads the nodes' residents (nodes.py, nodeserver.py), then
//               starts the server; from there, a message is read, run and
//               answered by the threads here.
//   NodeClient  the engine's side of a tcp:// mailbox.
//   StepHost    (#409) the engine's side of the step graph's mailboxes
//               (stepgraph.py): a thread that waits for each host node's
//               flag from the GPU, runs the node (its native package here,
//               or a node server's through a NodeClient) on the pinned
//               inputs, writes the reply into the pinned outbox and raises
//               the flag back, timing each step of the crossing.
//
// The wire format, both ways, little-endian (tcp:// mailboxes):
//
//   request   "VXN1"  u32 node  u32 count  count x tensor
//   reply     "VXR2"  u32 status (0: ok)   then, ok: u32 count, count x tensor,
//                                                    i64 read_ns, i64 run_ns
//                                          else: u32 length, the error text
//   tensor    u8 dtype (c10::ScalarType)  u8 ndim  ndim x i64 size  raw bytes
//
// A node is called with the message's tensors and then its residents, the
// tensors it reads that never travel (the banks of an experts node, the
// table of an n-gram node), and replies with the package's outputs. The
// reply says how long the server took to read the request (from its first
// byte) and to run the node, so the caller can tell the wire from the
// work (#409). Node kEchoNode replies with the request's tensors: the
// round trip with no work, for measuring the wire.

#include <torch/extension.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace {

constexpr uint32_t kMaxTensors = 64;
constexpr uint32_t kEchoNode = 0xFFFFFFFF;
constexpr uint8_t kMaxDims = 8;
constexpr int64_t kMaxBytes = int64_t(1) << 32;  // a message's tensor; a decode step's are kilobytes

void read_exact(int fd, void* buf, size_t n) {
  auto* p = static_cast<char*>(buf);
  while (n > 0) {
    ssize_t r = ::recv(fd, p, n, 0);
    if (r <= 0) throw std::runtime_error("connection closed");
    p += r;
    n -= static_cast<size_t>(r);
  }
}

void write_exact(int fd, const void* buf, size_t n) {
  auto* p = static_cast<const char*>(buf);
  while (n > 0) {
    ssize_t w = ::send(fd, p, n, MSG_NOSIGNAL);
    if (w <= 0) throw std::runtime_error("connection closed");
    p += w;
    n -= static_cast<size_t>(w);
  }
}

template <typename T>
T read_value(int fd) {
  T v;
  read_exact(fd, &v, sizeof(v));
  return v;
}

template <typename T>
void write_value(int fd, T v) {
  write_exact(fd, &v, sizeof(v));
}

int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch()).count();
}

at::Tensor read_tensor(int fd) {
  uint8_t code = read_value<uint8_t>(fd);
  if (code >= static_cast<uint8_t>(c10::ScalarType::NumOptions)) throw std::runtime_error("not a dtype");
  auto dtype = static_cast<c10::ScalarType>(code);
  uint8_t ndim = read_value<uint8_t>(fd);
  if (ndim > kMaxDims) throw std::runtime_error("too many dimensions");
  std::vector<int64_t> sizes(ndim);
  int64_t numel = 1;
  for (auto& s : sizes) {
    s = read_value<int64_t>(fd);
    if (s < 0) throw std::runtime_error("a negative size");
    if (s != 0 && numel > kMaxBytes / s) throw std::runtime_error("a tensor larger than the limit");
    numel *= s;
  }
  // checked before allocating: the sizes come from the peer
  int64_t bytes = numel * static_cast<int64_t>(c10::elementSize(dtype));
  if (bytes > kMaxBytes) throw std::runtime_error("a tensor larger than the limit");
  auto t = at::empty(sizes, at::TensorOptions().dtype(dtype));
  read_exact(fd, t.data_ptr(), static_cast<size_t>(bytes));
  return t;
}

void write_tensor(int fd, const at::Tensor& t0) {
  auto t = t0.contiguous().cpu();
  write_value<uint8_t>(fd, static_cast<uint8_t>(t.scalar_type()));
  write_value<uint8_t>(fd, static_cast<uint8_t>(t.dim()));
  for (auto s : t.sizes()) write_value<int64_t>(fd, s);
  write_exact(fd, t.data_ptr(), static_cast<size_t>(t.numel() * t.element_size()));
}

struct Node {
  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> loader;
  std::vector<at::Tensor> residents;
  std::mutex mu;  // one call of a node at a time: its package's buffers are its own
};

}  // namespace

// NodeServer serves nodes (by index: the order they were given in) on a
// port, until stopped. The accept loop and each connection run on their
// own threads, without the GIL.
// Connections are the connections a server is answering, each on a
// thread of its own. stop() ends them and waits for their threads: a
// detached thread would go on using the server after it is freed.
class Connections {
 public:
  // start runs serve(conn) on a thread; conn is closed when it returns.
  void start(int conn, std::function<void(int)> serve) {
    std::lock_guard<std::mutex> lock(mu_);
    if (stopped_) {
      ::close(conn);
      return;
    }
    open_.insert(conn);
    threads_.emplace_back([this, conn, serve = std::move(serve)] {
      serve(conn);
      std::lock_guard<std::mutex> lock(mu_);  // closed under the lock: stop() never shuts down a reused fd
      open_.erase(conn);
      ::close(conn);
    });
  }

  void stop() {
    std::vector<std::thread> threads;
    {
      std::lock_guard<std::mutex> lock(mu_);
      stopped_ = true;
      for (int fd : open_) ::shutdown(fd, SHUT_RDWR);
      threads.swap(threads_);
    }
    // a connection may be waiting on the GIL (a server whose nodes run in
    // Python): let it have it while we wait
    std::optional<pybind11::gil_scoped_release> release;
    if (PyGILState_Check()) release.emplace();
    for (auto& t : threads) t.join();
  }

 private:
  std::mutex mu_;
  bool stopped_ = false;
  std::set<int> open_;
  std::vector<std::thread> threads_;
};

class NodeServer {
 public:
  NodeServer(int64_t port, const std::vector<std::string>& packages,
             const std::vector<std::vector<at::Tensor>>& residents) {
    TORCH_CHECK(packages.size() == residents.size(), "a residents list per package");
    for (size_t i = 0; i < packages.size(); ++i) {
      auto node = std::make_unique<Node>();
      node->loader = std::make_unique<torch::inductor::AOTIModelPackageLoader>(packages[i]);
      node->residents = residents[i];
      nodes_.push_back(std::move(node));
    }
    fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    TORCH_CHECK(fd_ >= 0, "node server: socket");
    int one = 1;
    ::setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);  // other machines' engines call it: a private network
    addr.sin_port = htons(static_cast<uint16_t>(port));
    TORCH_CHECK(::bind(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0,
                "node server: bind to port ", port, ": ", std::strerror(errno));
    TORCH_CHECK(::listen(fd_, 16) == 0, "node server: listen");
    socklen_t len = sizeof(addr);
    ::getsockname(fd_, reinterpret_cast<sockaddr*>(&addr), &len);
    port_ = ntohs(addr.sin_port);
    accept_ = std::thread([this] { accept_loop(); });
  }

  ~NodeServer() { stop(); }

  int64_t port() const { return port_; }

  int64_t messages() const { return messages_.load(); }

  void stop() {
    if (stopped_.exchange(true)) return;
    ::shutdown(fd_, SHUT_RDWR);
    ::close(fd_);
    if (accept_.joinable()) accept_.join();
    connections_.stop();
  }

 private:
  void accept_loop() {
    while (!stopped_) {
      int conn = ::accept(fd_, nullptr, nullptr);
      if (conn < 0) {
        if (stopped_) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));  // out of fds, say: not a spin
        continue;
      }
      int one = 1;
      ::setsockopt(conn, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));  // a message is small and waited on
      connections_.start(conn, [this](int c) { serve(c); });
    }
  }

  void serve(int conn) {
    try {
      for (;;) {
        char magic[4];
        read_exact(conn, magic, 4);
        const int64_t t0 = now_ns();
        if (std::memcmp(magic, "VXN1", 4) != 0) throw std::runtime_error("not a node message");
        uint32_t index = read_value<uint32_t>(conn);
        uint32_t count = read_value<uint32_t>(conn);
        if (count > kMaxTensors) throw std::runtime_error("too many tensors");
        std::vector<at::Tensor> inputs;
        for (uint32_t i = 0; i < count; ++i) inputs.push_back(read_tensor(conn));
        const int64_t t1 = now_ns();
        std::vector<at::Tensor> outputs;
        std::string error;
        try {
          if (index == kEchoNode) {
            outputs = inputs;
          } else {
            if (index >= nodes_.size()) throw std::runtime_error("no such node");
            Node& node = *nodes_[index];
            inputs.insert(inputs.end(), node.residents.begin(), node.residents.end());
            std::lock_guard<std::mutex> lock(node.mu);
            outputs = node.loader->run(inputs);
          }
        } catch (const std::exception& e) {
          error = e.what();
        }
        const int64_t t2 = now_ns();
        write_exact(conn, "VXR2", 4);
        if (!error.empty()) {
          write_value<uint32_t>(conn, 1);
          write_value<uint32_t>(conn, static_cast<uint32_t>(error.size()));
          write_exact(conn, error.data(), error.size());
          continue;
        }
        write_value<uint32_t>(conn, 0);
        write_value<uint32_t>(conn, static_cast<uint32_t>(outputs.size()));
        for (const auto& t : outputs) write_tensor(conn, t);
        write_value<int64_t>(conn, t1 - t0);
        write_value<int64_t>(conn, t2 - t1);
        messages_++;
      }
    } catch (const std::exception&) {
      // the peer went away, or sent something that is not a message: drop it
    }
  }

  std::vector<std::unique_ptr<Node>> nodes_;
  int fd_ = -1;
  int64_t port_ = 0;
  std::thread accept_;
  Connections connections_;
  std::atomic<bool> stopped_{false};
  std::atomic<int64_t> messages_{0};
};

// NodeClient calls a node server's nodes: one connection, kept, one call
// at a time; the engine's side of a tcp:// mailbox.
// A node client is one connection to a node server. A call that fails
// (the server gone, or silent for kCallTimeoutSec) fails the step and
// closes the connection, and the next call connects again: a hung or
// restarted server must not wedge the step host that waits on it.
class NodeClient {
 public:
  NodeClient(const std::string& host, int64_t port) : host_(host), port_(port) { connect(); }

  ~NodeClient() {
    if (fd_ >= 0) ::close(fd_);
  }

  std::vector<at::Tensor> call(int64_t node, const std::vector<at::Tensor>& inputs) {
    std::lock_guard<std::mutex> lock(mu_);
    if (fd_ < 0) connect();
    try {
      return exchange(node, inputs);
    } catch (...) {
      ::close(fd_);  // mid-message, or the server's state unknown: start over
      fd_ = -1;
      throw;
    }
  }

  // The last call's server times: reading the request, running the node (ns).
  std::pair<int64_t, int64_t> server_times() const { return {read_ns_, run_ns_}; }

 private:
  // a long prefill's experts on a CPU take seconds; a server silent for
  // two minutes is gone
  static constexpr int kCallTimeoutSec = 120;

  void connect() {
    const std::string& host = host_;
    const int64_t port = port_;
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* found = nullptr;
    int rc = ::getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &found);
    TORCH_CHECK(rc == 0, "node client: resolving ", host, ": ", ::gai_strerror(rc));
    for (addrinfo* a = found; a != nullptr; a = a->ai_next) {
      fd_ = ::socket(a->ai_family, a->ai_socktype, a->ai_protocol);
      if (fd_ >= 0 && ::connect(fd_, a->ai_addr, a->ai_addrlen) == 0) break;
      if (fd_ >= 0) ::close(fd_);
      fd_ = -1;
    }
    ::freeaddrinfo(found);
    TORCH_CHECK(fd_ >= 0, "node client: cannot connect to ", host, ":", port);
    int one = 1;
    ::setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    timeval timeout{kCallTimeoutSec, 0};
    ::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    ::setsockopt(fd_, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
  }

  std::vector<at::Tensor> exchange(int64_t node, const std::vector<at::Tensor>& inputs) {
    write_exact(fd_, "VXN1", 4);
    write_value<uint32_t>(fd_, static_cast<uint32_t>(node));
    write_value<uint32_t>(fd_, static_cast<uint32_t>(inputs.size()));
    for (const auto& t : inputs) write_tensor(fd_, t);
    char magic[4];
    read_exact(fd_, magic, 4);
    TORCH_CHECK(std::memcmp(magic, "VXR2", 4) == 0, "node client: not a node server reply (VXR2)");
    if (read_value<uint32_t>(fd_) != 0) {
      uint32_t length = read_value<uint32_t>(fd_);
      std::string error(length, '\0');
      read_exact(fd_, error.data(), length);
      TORCH_CHECK(false, "node ", node, ": ", error);
    }
    uint32_t count = read_value<uint32_t>(fd_);
    TORCH_CHECK(count <= kMaxTensors, "node client: too many tensors");
    std::vector<at::Tensor> outputs;
    for (uint32_t i = 0; i < count; ++i) outputs.push_back(read_tensor(fd_));
    read_ns_ = read_value<int64_t>(fd_);
    run_ns_ = read_value<int64_t>(fd_);
    return outputs;
  }

  std::string host_;
  int64_t port_;
  int fd_ = -1;
  std::mutex mu_;
  int64_t read_ns_ = 0, run_ns_ = 0;
};

namespace {

void spin_pause() {
#if defined(__x86_64__)
  _mm_pause();
#else
  std::this_thread::yield();
#endif
}

}  // namespace

// StepHost answers the step graph's host nodes, in step order, once per
// step. Host node i owns two int32 flags in pinned memory the GPU can
// see: flags[2i] (to the CPU: its inputs are in the pinned inboxes) and
// flags[2i+1] (from the CPU: the reply is in the pinned outbox). Each
// node is answered by its own native package (local: residents passed by
// address, as a node server runs it) or by a node server (remote: a
// NodeClient and the server's node index). Per node and step it records,
// in steady-clock ns: when it saw the GPU's flag, when the reply was in
// the outbox, and when it raised the flag back; for a remote node also
// the server's own times (reading the request, running the node).
class StepHost {
 public:
  explicit StepHost(at::Tensor flags) : flags_(std::move(flags)) {
    TORCH_CHECK(flags_.dtype() == at::kInt && flags_.is_contiguous() && flags_.is_pinned(),
                "step host: the flags must be pinned contiguous int32");
  }

  ~StepHost() { stop(); }

  void add_local(const std::vector<at::Tensor>& inputs, const at::Tensor& outbox, const std::string& package,
                 const std::vector<at::Tensor>& residents) {
    auto box = make_box(inputs, outbox);
    box->loader = std::make_unique<torch::inductor::AOTIModelPackageLoader>(package);
    box->residents = residents;
    boxes_.push_back(std::move(box));
  }

  void add_remote(const std::vector<at::Tensor>& inputs, const at::Tensor& outbox,
                  std::shared_ptr<NodeClient> client, int64_t node) {
    auto box = make_box(inputs, outbox);
    box->client = std::move(client);
    box->node = node;
    boxes_.push_back(std::move(box));
  }

  void start() {
    TORCH_CHECK(!thread_.joinable(), "step host: already started");
    TORCH_CHECK(flags_.numel() >= 2 * static_cast<int64_t>(boxes_.size()), "step host: two flags per host node");
    times_ = at::zeros({static_cast<int64_t>(boxes_.size()), 5}, at::kLong);
    thread_ = std::thread([this] { loop(); });
  }

  // One step: the GPU is about to raise the first flag.
  void go() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      ++steps_requested_;
    }
    cv_.notify_one();
  }

  // After the GPU finished the step: the error of the step's failed node
  // ("" if none). A failure raises every from-CPU flag, so the GPU never
  // waits forever; the caller then clears the flags before the next step.
  std::string take_error() {
    std::lock_guard<std::mutex> lock(mu_);
    std::string e;
    std::swap(e, error_);
    return e;
  }

  // [nodes, 5] int64: per host node of the last step, when the GPU's flag
  // was seen, when the reply was ready, when the flag was raised back
  // (steady-clock ns), and for a remote node the server's read and run
  // times (ns; 0 for a local node).
  at::Tensor times() const { return times_; }

  bool running() const { return thread_.joinable(); }

  void stop() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stopping_ = true;
    }
    halt_ = true;  // also ends a wait on a GPU flag that will never come (a step the GPU abandoned)
    cv_.notify_one();
    if (thread_.joinable()) thread_.join();
  }

 private:
  struct Box {
    std::vector<at::Tensor> inputs;  // pinned, filled by the GPU
    at::Tensor outbox;               // pinned, read by the GPU
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> loader;
    std::vector<at::Tensor> residents;
    std::shared_ptr<NodeClient> client;
    int64_t node = 0;
  };

  std::unique_ptr<Box> make_box(const std::vector<at::Tensor>& inputs, const at::Tensor& outbox) {
    for (const auto& t : inputs) TORCH_CHECK(t.is_pinned(), "step host: inputs must be pinned");
    TORCH_CHECK(outbox.is_pinned() && outbox.is_contiguous(), "step host: the outbox must be pinned and contiguous");
    auto box = std::make_unique<Box>();
    box->inputs = inputs;
    box->outbox = outbox;
    return box;
  }

  void loop() {
    volatile int32_t* flag = flags_.data_ptr<int32_t>();
    int64_t* t = times_.data_ptr<int64_t>();
    for (;;) {
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [this] { return stopping_ || steps_done_ < steps_requested_; });
        if (stopping_) return;
      }
      size_t i = 0;
      try {
        for (; i < boxes_.size(); ++i) {
          Box& box = *boxes_[i];
          while (flag[2 * i] != 1) {
            if (halt_.load(std::memory_order_relaxed)) return;
            spin_pause();
          }
          flag[2 * i] = 0;
          t[5 * i] = now_ns();
          std::vector<at::Tensor> out;
          if (box.client) {
            out = box.client->call(box.node, box.inputs);
            auto [read_ns, run_ns] = box.client->server_times();
            t[5 * i + 3] = read_ns;
            t[5 * i + 4] = run_ns;
          } else {
            std::vector<at::Tensor> args(box.inputs);
            args.insert(args.end(), box.residents.begin(), box.residents.end());
            out = box.loader->run(args);
          }
          TORCH_CHECK(!out.empty(), "host node ", i, ": no reply");
          box.outbox.copy_(out[0].view(box.outbox.sizes()));
          t[5 * i + 1] = now_ns();
          t[5 * i + 2] = now_ns();  // as the flag goes up: before it, so the step's reader never sees it unwritten
          std::atomic_thread_fence(std::memory_order_seq_cst);  // the reply (and the times) before the flag
          flag[2 * i + 1] = 1;
        }
      } catch (const std::exception& e) {
        {
          std::lock_guard<std::mutex> lock(mu_);
          error_ = "host node " + std::to_string(i) + ": " + e.what();
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        for (size_t k = 0; k < boxes_.size(); ++k) flag[2 * k + 1] = 1;  // release the GPU
      }
      std::lock_guard<std::mutex> lock(mu_);
      ++steps_done_;
    }
  }

  at::Tensor flags_;
  at::Tensor times_;
  std::vector<std::unique_ptr<Box>> boxes_;
  std::thread thread_;
  std::mutex mu_;
  std::condition_variable cv_;
  int64_t steps_requested_ = 0, steps_done_ = 0;
  bool stopping_ = false;
  std::atomic<bool> halt_{false};
  std::string error_;
};

void register_node_server(pybind11::module& m) {
  pybind11::class_<NodeClient, std::shared_ptr<NodeClient>>(m, "NodeClient")
      .def(pybind11::init<const std::string&, int64_t>())
      .def("call", &NodeClient::call, pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("server_times", &NodeClient::server_times);
  pybind11::class_<NodeServer>(m, "NodeServer", pybind11::dynamic_attr())
      .def(pybind11::init<int64_t, const std::vector<std::string>&, const std::vector<std::vector<at::Tensor>>&>(),
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("port", &NodeServer::port)
      .def("messages", &NodeServer::messages)
      .def("stop", &NodeServer::stop, pybind11::call_guard<pybind11::gil_scoped_release>());
  pybind11::class_<StepHost>(m, "StepHost")
      .def(pybind11::init<at::Tensor>())
      .def("add_local", &StepHost::add_local, pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("add_remote", &StepHost::add_remote)
      .def("start", &StepHost::start)
      .def("go", &StepHost::go)
      .def("take_error", &StepHost::take_error)
      .def("times", &StepHost::times)
      .def("running", &StepHost::running)
      .def("stop", &StepHost::stop, pybind11::call_guard<pybind11::gil_scoped_release>());
}
