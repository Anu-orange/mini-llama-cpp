// Copyright (c) 2026 yus3nable
// SPDX-License-Identifier: MIT

#include "mini_llama/thread_pool.h"

#include <algorithm>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace mini_llama {

// User-selected ParallelFor override; zero uses the hardware concurrency
// default.
static int g_thread_count = 0;

int GetThreadCount() {
	if (g_thread_count > 0) {
		return g_thread_count;
	}
	// 硬件的并发数
	unsigned int hw = std::thread::hardware_concurrency();
	return hw > 0 ? static_cast<int>(hw) : 4;
}

void SetThreadCount(int n) { g_thread_count = n > 0 ? n : 0; }

/**
 * @brief 多线程按照调用方给定的任务数进行拆分，
 * 也就是多个线程每个线程拿一部分数据，然后由该函数中的函数对象进行处理
 * 
 * @param n 
 * @param fn 
 */
void ParallelFor(int n, const std::function<void(int begin, int end)>& fn) {
	if (n <= 0) {
		return;
	}

	// 获取线程数
	int n_threads = GetThreadCount();

	// Only parallelize if each thread gets a meaningful chunk of work.

	// 每个线程最小的任务数
	constexpr int kMinChunk = 16;
	if (n < n_threads * kMinChunk) {
		// 如果任务总数小于总线程数乘以每个线程的最小任务数
		// 则直接调用一次函数对象处理0，n中间这n个任务
		fn(0, n);
		return;
	}

	// 如果线程数要是大于了任务数的话，那么就将线程数限制为任务数
	// 原因是多出的那部分线程属于是白创建了，都没有任务去执行
	if (n_threads > n) {
		n_threads = n;
	}

	std::vector<std::thread> threads;
	threads.reserve(n_threads);
	std::exception_ptr first_exception = nullptr;
	std::mutex exception_mutex;

	// 每个线程至少分配的任务数
	int chunk = n / n_threads;
	// 剩余的任务数
	int remainder = n % n_threads;

	int start = 0;
	for (int t = 0; t < n_threads; ++t) {
		// 这里为每个chunk加1，是需要将剩余的remainder个任务数平分到前remainder个线程上，前remainder个线程
		// 在处理的任务数chunk上多加1个
		int count = chunk + (t < remainder ? 1 : 0);
		int end = start + count;
		if (count > 0) {
			// 在threads的末尾位置原地构造一个线程对象，std::thread类的构造函数接受一个lamada表达式
			// 把这个线程对象创建出来
			threads.emplace_back(
			    [start, end, &fn, &first_exception, &exception_mutex]() {
				    try {
					    fn(start, end);
				    } catch (...) {
						// 当前对异常互斥量进行上锁，确保只有一个线程来写异常指针
					    std::lock_guard<std::mutex> lock(exception_mutex);
					    if (!first_exception) {
							// 当前异常指针为空时才写入异常信息，也就是说非首个产生异常的线程将不会写进来
							// 因为前面有线程写了之后，该指针将为非空，则不会进入该if条件内部
						    first_exception = std::current_exception();
					    }
				    }
			    });
		}
		start = end;
	}

	// 当前线程被阻塞，等到threads中所有线程全部执行完毕后，当前线程将继续执行
	for (auto& t : threads) {
		t.join();
	}

	if (first_exception) {
		std::rethrow_exception(first_exception);
	}
}

}  // namespace mini_llama
