#include "image_retrieval/concurrency/thread_pool.h"

namespace image_retrieval {
namespace concurrency {

ThreadPool::~ThreadPool() {
  {
    absl::MutexLock l(&mu_);
    for (size_t i = 0; i < threads_.size(); i++) {
      queue_.push(nullptr);  // Shutdown signal.
    }
  }
  for (auto &t : threads_) {
    t.join();
  }
}

void ThreadPool::Schedule(std::function<void()> func) {
  assert(func != nullptr);
  absl::MutexLock l(&mu_);
  queue_.push(std::move(func));
}

void ThreadPool::WorkLoop() {
  while (true) {
    std::function<void()> func;
    {
      absl::MutexLock l(&mu_);
      mu_.Await(absl::Condition(this, &ThreadPool::WorkAvailable));
      func = std::move(queue_.front());
      queue_.pop();
    }
    if (func == nullptr) {  // Shutdown signal.
      break;
    }
    func();
  }
}

}  // namespace concurrency
}  // namespace image_retrieval
