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

// A node server (#394): dataflow nodes, as AOTInductor packages, answering
// their mailboxes over TCP with no Python between messages. Python builds
// the packages and loads the nodes' residents (nodes.py, nodeserver.py),
// then starts the server; from there, a message is read, run and answered
// by the threads here.
//
// The wire format, both ways, little-endian (tcp:// mailboxes):
//
//   request   "VXN1"  u32 node  u32 count  count x tensor
//   reply     "VXR1"  u32 status (0: ok)   then, ok: u32 count, count x tensor
//                                          else: u32 length, the error text
//   tensor    u8 dtype (c10::ScalarType)  u8 ndim  ndim x i64 size  raw bytes
//
// A node is called with the message's tensors and then its residents, the
// tensors it reads that never travel (the banks of an experts node, the
// table of an n-gram node), and replies with the package's outputs.

#include <torch/extension.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
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

namespace {

constexpr uint32_t kMaxTensors = 64;
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
        if (std::memcmp(magic, "VXN1", 4) != 0) throw std::runtime_error("not a node message");
        uint32_t index = read_value<uint32_t>(conn);
        uint32_t count = read_value<uint32_t>(conn);
        if (count > kMaxTensors) throw std::runtime_error("too many tensors");
        std::vector<at::Tensor> inputs;
        for (uint32_t i = 0; i < count; ++i) inputs.push_back(read_tensor(conn));
        std::vector<at::Tensor> outputs;
        std::string error;
        try {
          if (index >= nodes_.size()) throw std::runtime_error("no such node");
          Node& node = *nodes_[index];
          inputs.insert(inputs.end(), node.residents.begin(), node.residents.end());
          std::lock_guard<std::mutex> lock(node.mu);
          outputs = node.loader->run(inputs);
        } catch (const std::exception& e) {
          error = e.what();
        }
        write_exact(conn, "VXR1", 4);
        if (!error.empty()) {
          write_value<uint32_t>(conn, 1);
          write_value<uint32_t>(conn, static_cast<uint32_t>(error.size()));
          write_exact(conn, error.data(), error.size());
          continue;
        }
        write_value<uint32_t>(conn, 0);
        write_value<uint32_t>(conn, static_cast<uint32_t>(outputs.size()));
        for (const auto& t : outputs) write_tensor(conn, t);
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

void register_node_server(pybind11::module& m) {
  pybind11::class_<NodeServer>(m, "NodeServer", pybind11::dynamic_attr())
      .def(pybind11::init<int64_t, const std::vector<std::string>&, const std::vector<std::vector<at::Tensor>>&>(),
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("port", &NodeServer::port)
      .def("messages", &NodeServer::messages)
      .def("stop", &NodeServer::stop, pybind11::call_guard<pybind11::gil_scoped_release>());
}
