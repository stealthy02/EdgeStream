#include "thread_safe_queue.h"

#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

namespace {

using namespace std::chrono_literals;

bool expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        return false;
    }
    return true;
}

bool producer_blocks_until_space_is_available() {
    ThreadSafeQueue<int> queue(1);
    queue.push(1);
    std::atomic<bool> pushed{false};

    std::thread producer([&] {
        queue.push(2);
        pushed.store(true, std::memory_order_release);
    });

    std::this_thread::sleep_for(50ms);
    bool ok = expect(!pushed.load(std::memory_order_acquire),
                     "producer must block while the bounded queue is full");

    int value = 0;
    ok &= expect(queue.pop(value) && value == 1,
                 "consumer must receive the item already in the queue");
    producer.join();
    ok &= expect(pushed.load(std::memory_order_acquire),
                 "blocked producer must resume after space is available");
    ok &= expect(queue.pop(value) && value == 2,
                 "resumed producer item must be preserved");
    return ok;
}

bool consumer_waits_until_item_is_available() {
    ThreadSafeQueue<int> queue(1);
    std::atomic<bool> popped{false};
    int value = 0;

    std::thread consumer([&] {
        popped.store(queue.pop(value), std::memory_order_release);
    });

    std::this_thread::sleep_for(50ms);
    bool ok = expect(!popped.load(std::memory_order_acquire),
                     "consumer must wait while the queue is empty");
    queue.push(42);
    consumer.join();
    ok &= expect(popped.load(std::memory_order_acquire) && value == 42,
                 "waiting consumer must receive the produced item");
    return ok;
}

bool stop_drains_items_then_reports_end() {
    ThreadSafeQueue<int> queue(2);
    queue.push(10);
    queue.push(20);
    queue.stop();

    int value = 0;
    bool ok = expect(queue.pop(value) && value == 10,
                     "stop must not discard the first queued item");
    ok &= expect(queue.pop(value) && value == 20,
                 "stop must not discard the remaining queued item");
    ok &= expect(!queue.pop(value),
                 "consumer must exit after stopped queue is drained");
    return ok;
}

bool stopped_producer_does_not_remain_blocked() {
    // Leak only on failure: the current implementation may leave push blocked,
    // and destroying its queue while that thread is waiting would be unsafe.
    auto* queue = new ThreadSafeQueue<int>(1);
    queue->push(1);
    std::atomic<bool> finished{false};
    std::thread producer([queue, &finished] {
        queue->push(2);
        finished.store(true, std::memory_order_release);
    });

    std::this_thread::sleep_for(50ms);
    queue->stop();
    const auto deadline = std::chrono::steady_clock::now() + 250ms;
    while (!finished.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }

    if (!finished.load(std::memory_order_acquire)) {
        std::cerr << "FAIL: stop must release a producer blocked by a full queue\n";
        producer.detach();
        return false;
    }

    producer.join();
    delete queue;
    return true;
}

}  // namespace

int main() {
    bool ok = true;
    ok &= producer_blocks_until_space_is_available();
    ok &= consumer_waits_until_item_is_available();
    ok &= stop_drains_items_then_reports_end();
    ok &= stopped_producer_does_not_remain_blocked();

    if (ok) {
        std::cout << "ThreadSafeQueue regression tests passed\n";
        return 0;
    }
    return 1;
}
