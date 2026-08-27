#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>

template <typename T>
class ThreadSafeQueue {
public:
    explicit ThreadSafeQueue(size_t max_size) : max_size_(max_size) {}

    bool push(T item) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_not_full_.wait(lock, [this] { return queue_.size() < max_size_ || is_stopped_; });
        if (is_stopped_) {
            return false;
        }
        queue_.push(item);
        cv_not_empty_.notify_one();
        return true;
    }

    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mtx_);
        // 等待条件：队列有数据，或者队列被要求停止
        cv_not_empty_.wait(lock, [this] { 
            return queue_.size() > 0 || is_stopped_; 
        });

        // 醒来后检查：如果是被停止信号唤醒，且队列里已经没数据了
        if (queue_.empty() && is_stopped_) {
            return false; 
        }

        item = queue_.front();
        queue_.pop();
        cv_not_full_.notify_one();
        return true;
    }

    int size(){
        std::unique_lock<std::mutex> lock(mtx_);
        return queue_.size();
    }
    bool empty(){
        std::unique_lock<std::mutex> lock(mtx_);
        return queue_.size() == 0;
    }
    void stop() {
        std::unique_lock<std::mutex> lock(mtx_);
        is_stopped_ = true;
        cv_not_empty_.notify_all(); // 唤醒所有卡在 pop 处的消费者
        cv_not_full_.notify_all();  // 防止生产者卡在 push 处
    }
private:
    std::queue<T> queue_;
    std::mutex mtx_;
    std::condition_variable cv_not_empty_;
    std::condition_variable cv_not_full_;
    size_t max_size_;
    bool is_stopped_ = false;
};