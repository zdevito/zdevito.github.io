---
layout: post
title:  "A guide to PyTorch's CUDA Caching Allocator"
date:   2022-08-03 21:55:37 -0700
categories: ""
---

A guide to PyTorch's CUDA Caching Allocator
===========================================

The goal of the CUDA caching allocator in PyTorch is to reach a steady state where the program runs without needing to request new memory from CUDA using `cudaMalloc` and `cudaFree`. PyTorch relies on the CPU execution running ahead of GPU execution to hide the latency of the Python interpreter behind the more expensive CUDA operations. But these memory APIs, especially cudaFree, introduce synchronization which interfere with this process.

To accomplish its goal, the caching allocator requests blocks of memory from CUDA and figures out ways to split up and reuse these blocks without returning them to CUDA.

Why not just request all GPU memory and manage it inside PyTorch? PyTorch is not the only library to use the CUDA APIs. Many programs will use cuBlas, cuDNN, and NCCL all of which do some allocations on their own, and users might mix PyTorch with other CUDA-accelerated libraries we do not know about. Instead, if PyTorch ever uses N bytes of memory at one point, we will continue to to keep that N bytes cached until a user specifically frees it via `torch.cuda.memory.empty_cache()`.

Under normal circumstances, the allocator achieves these goals and mostly lives in the background. But sometimes a particular set of adversarial allocations might prevent it from reaching a steady state. Or maybe a program runs out of memory for good reasons but it is hard to read the statistics. In these cases, it is helpful to understand more about what the allocator is doing to debug the issue.

This document gives an overview with pseudocode for how the allocator behaves as of August 2022.

Allocation
----------

The overall approach to allocating memory is pretty simple. We maintain a cache of allocations (Blocks) that we have previously gotten from CUDA that we will attempt to reuse:

    struct Block {
        void* addr;
        size_t size;
        bool active;
        set<Block>* pool;
        // for splitting and merging blocks.
        Block* prev, next;
    };

    set<Block> small_pool;
    set<Block> large_pool;

For < 1MB "small" tensors, we use a separate pool to avoid fragmenting the larger pool.
At a high level, we try to find a free block from this pool, but ask CUDA for more memory otherwise:

    Block malloc(int device, size_t size, cudaStream_t stream) {

        process_cross_stream_delayed_free()
        size = round_size(size); 
        pool = size < 1MB ? small_pool : large_pool;

        Block block = <find and remove the smallest block in the pool on the same stream that is big enough to fit size>
        if (<block not found>) {
            block = alloc_new_block(size);
            block.pool = pool
        }
        block = maybe_split_block(pool, block);
        return block;
    }

The allocation starts with some cleanup ([Streams](#streams-and-freeing-memory)) and  [allocation rounding](#allocation-rounding) of the size to be allocated. It then uses a best fit strategy finding the smallest block that fits for the allocation. If none is already in the cache, then we are not yet in a good steady state and ask CUDA for more memory.

Finally, the block of allocated memory we choose may be significantly larger than what was requested. This can happen if we are reusing a previous bigger allocation for a smaller one, or because we always allocate 2MB for the small pool and 20MB for the large pool to avoid calling cudaMalloc too frequently. In these cases, `maybe_split_block` can split the block so the rest can be used for other allocations.

Blocks are allocated _per CUDA stream_. Each CUDA stream will have its own cache of allocations, with rebalancing done only by emptying all the caches in low-memory conditions.

Allocation Rounding
-------------------

Each allocation is rounded up to make it less likely to end up with fragmentation from oddly shaped allocations. Rounding happens twice. First on the size being requested, and again if we are requesting a (possibly bigger) block from CUDA. The requested size is immediately rounded up using `round_size`:

    size_t round_size(size_t size) {
        if (<default>) {
            return <round to a multiple of 512B>
        } else { // via configuration using CUDA_PYTORCH_CUDA_ALLOC_CONF
            <round to a power of 2 with N divisions between, minimum size is 512>
          // For example, if we need to allocate 1200 and number of divisions is 4,
          // the size 1200 lies between 1024 and 2048 and if we do 4 divisions between
          // them, the values are 1024, 1280, 1536, and 1792. So the function will
          // return 1280 as the nearest ceiling of power-2 divison.
        }
    }

The environment variable `CUDA_PYTORCH_CUDA_ALLOC_CONF=roundup_power2_divisions:N` makes the rounding more aggressive to try to avoid situations where small changes in batch size or sequence length will cause different-sized allocations each time, making it harder to reach a steady state. Some memory is lost to this rounding (N=1 will on average waste 1/4th of each allocation but N=2 will only waste 1/8th). Since many models do not use varying sizes, it is disabled by default.



Asking CUDA For More Memory
---------------------------

A second set of rounding rules applying when we need more memory from CUDA. In this circumstance, we only ask CUDA for a multiple of 2MB at its smallest, expecting to split the block up for smaller allocations. For large sizes this rounding happens in 2MB chunks:

    Block alloc_new_block(size_t size) {
        // when we allocate a new block, we allocate it bigger than the requested object
        // in order to be able to reuse this allocation by splitting it.
        if (size < 1MB) {
            allocation_size = 2MB;
        } else if (size < 10MB) {
            allocation_size = 20MB
        } else {
            allocation_size = <size rounded a multiple of 2MB>
        }
        memory = cudaMalloc(allocation_size)
        if (<cuda is out of memory>) {
            return_our_possibly_fragmented_memory_to_cuda();
            memory = cudaMalloc(allocation_size);
            if (<cuda still out of memory>) {
                <throw an out of memory exception>
            }
        }
        return Block(memory)
    }


Allocating a new block from memory is also when we discover that CUDA is out of memory (an OOM). In this situation we will try to free up cached but unused memory. Before we see how this works, we need to see how Blocks are split and recombined.



Block Splitting/Merging
-----------------------
We ask CUDA for larger blocks of memory than we need for small (<1MB) and medium sized (1--10MB) allocations with the intention of using this block for multiple allocations:

    Block maybe_split_block(Pool pool, size_t size, Block block) {
        remaining = block.size - size
        should_split = (size < 1MB && remaining > 512B) || (size >= 1MB && remaining > 1MB)
        if (!should_split) {
            return block;
        }
        block, rest = <split the first 'size' bytes into in new block, leaving a remaining block>
        pool.add(rest)
    }

The remaining block is added to the pool and can be used for another allocation. Eventually a `free` will return the block to the cached pool.
To avoid fragmentation of the big slab of memory, when a block is returned, it will merge with its neighbors if they are also free, creating a larger slab.


    void return_block_to_reserved_memory(Block block) {
        if (<block was split from a larger block>) {
            if (<sucessor or predecessor of split block is free>) {
                <merge with free sucessor and predecessor>
            } else {
                pool.add(block)
            }
        } else {
            pool.add(block)
        }
    }


Streams and freeing memory
--------------------------

Events on a GPU are run in the order they are issued to a CUDA stream. Because we want the CPU to run ahead of the GPU, we also treat memory allocation and memory freeing as an event that happens in order on a stream. This lets us re-use an allocation on the same stream it was just freed on, even if we do not wait for the kernel using that allocation to completely finish before scheduling a kernel that will use the new allocation. This works because we know the GPU must finish the kernel with the last use of the old allocation before starting the kernel that uses the new one.

There are no such ordering guarantees between streams, so extra care has to happen when a tensor `A` is allocated on one stream `creator` but used on another stream `user`. In these cases, users have to call `A.record_stream(user)` to let the allocator know about `A`'s use on `user`. During `free`, the allocator will only consider the block ready for reuse when all work that has been scheduled up to the point `A` became free on `user` is complete. This is done by recording an event on the `user` stream and only handing out `A`'s member after that event has elapsed from the perspective of the CPU:

    void free(Block* block) {
        if (<block A has been used on a stream 'user' that it was not 'created' on>) {
            <record a cuda event to the end of 'user' stream and save it with A>
            <defer freeing A until those events have elapsed>
            // condition is checked in process_cross_stream_delayed_free()
        } else {
            return_block_to_reserved_memory(block)
        }
    }

    // called right before malloc
    void process_cross_stream_delayed_free() {
        for all blocks waiting on events that have no remaining events {
            return_block_to_reserved_memory(block)
        }
    }


Notice how that even when a block is used on a different stream its memory is always returned to the stream where it was allocated.

Low memory conditions and cudaMalloc retry
------------------------------------------

The final piece of the allocator is how to recover during an OOM. This is referred to as a `cudaMalloc` retry in the statistics. When we have a lot of cached blocks it is possible that a lot of them are free but too small to fulfill a large allocation. This can be especially true if the allocation is a larger version (e.g. larger batch, longer sequence) of allocations from a previous step. The caching allocator cannot move these fragmented blocks around to construct a larger allocation. However, CUDA can move this memory around by changing page tables on the GPU. So our approach is to return these blocks to CUDA via cudaFree and try again:

    void return_our_possibly_fragmented_memory_to_cuda() {
        <wait for all cross stream events to complete>
        <for all free blocks in our reserved memory, cudaFree them>
    }

Since `cudaFree` synchronizes the device, this process is very expensive, so we use this time to also free any of the cross stream blocks we were waiting for as well.


At this point if we are out of memory, we raise an `OutOfMemoryError` exception (the OOM statistic). Under some circumstance, it might be the case that there is enough memory in the system to continue but we cannot free it. For instance, if there is a small tensor allocated in a big block of memory, we cannot free that block, wasting the rest of the block.

Something to keep in mind in the OOM situations is that the model is probably right in the middle of step, near the end of a forward pass before backward when we are keeping alive a lot of temporaries for the backward pass. So there may be a lot of temporary tensors allocated and taking up parts of blocks. It is possible if you were to back up to the beginning of a step with fewer live temporaries, emptying the allocator caches might be more successful, and it might succeed at the iteration because it will directly cudaMalloc tensors of the new (larger) size:

    retry = False
    try:
        step()
    except torch.cuda.OutOfMemoryError:
        retry = True
        # exception handlers hold on to the stack trace frames,
        # the memory of temporaries will still be alive until
        # we exit the exception handler
    if retry:
        torch.cuda.memory.empty_cache()
        step() # maybe we succeed now


The meaning of metrics
----------------------

The allocator provides a number of metrics, accessed through functions like `torch.cuda.memory_stats()` or `torch.cuda.memory_summary()`, that correspond to the state presented in the pseudo-code above.

* `allocated_bytes` - The sum of the size of all Blocks in the active state. This includes memory added by `round_size`, so for instance a 1 byte allocation will be counted at 512 bytes here. This does not include blocks waiting for to be freed in `process_cross_stream_delayed_free()`.
* `reserved_bytes` - The sum of the size of all Blocks, regardless of state.
* `inactive_split_bytes` -  The sum of the size of all Blocks in an inactive state that were split from a larger segment via `maybe_split_block`. This does not include blocks waiting for to be freed in `process_cross_stream_delayed_free()`. These are a potential source of fragmentation because they cannot be returned to CUDA during a cudaMalloc retry since another part of the block is still occupied.
* `cudaMalloc retries` - The number of times `return_our_possibly_fragmented_memory_to_cuda()` has been called. If this is being called frequently, it indicates that the allocated has failed to reach a goal state and is still regularly calling synchronous cudaFree operations.
* `CUDA OOMs` - The number of times the allocator has thrown a `OutOfMemoryError`. Most non-interactive programs will not recover from this, but Python notebooks might accumulate multiple OOMs.

What's next
-----------

To make it easier to understand precisely what happened that lead to an OOM, we've been developing [new tools](https://github.com/pytorch/pytorch/pull/82146) to visualize the state of memory.
It records Python stack traces with each allocation and keeps the last allocation that existed in each block, and can visualize them with [flamegraphs](https://www.brendangregg.com/flamegraphs.html). In the next post, I will show how to use the tool to debug memory issues.

