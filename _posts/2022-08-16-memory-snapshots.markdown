---
layout: post
title:  "Debugging PyTorch memory use with snapshots"
date:   2022-08-15 21:55:37 -0700
categories: ""
---

Debugging PyTorch memory use with snapshots
===========================================

In a [previous post](https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html), I gave a detailed guide about how PyTorch CUDA caching allocator hands out memory. To understand how it is working with your own training run or application, we also have developed new tools to visualize the state of allocated memory by generating and visualzing _memory snapshots_. These snapshots record the stack traces where memory was allocated and where it lives inside the state of the caching allocator. By visualizing these snapshots as [flamegraphs](https://www.brendangregg.com/flamegraphs.html), it can show where memory is being used at glance.


Generating snapshots
--------------------

First, we have to enable the recording of stack frame information for each allocation:

    import torch
    torch.cuda.memory._record_memory_history()

Recording these stack traces is pretty fast (~1us per allocation, a normal PyTorch kernel call takes at least 8 us), but we leave it off by default.  Once enabled we can allocate some memory, and take a snapshot:

    from torchvision.models import resnet18
    from pprint import pprint

    model = resnet18().cuda()
    input = torch.rand(1, 3, 224, 224).cuda()
    model.train()
    output = model(input)
    snapshot = torch.cuda.memory._snapshot()
    pprint(snapshot['segments'])

The snapshot records the entire allocator state and looks like:

    [{'active_size': 19398656,
    'address': 139896043864064,
    'allocated_size': 19398656,
    'blocks': [{'history': [{'addr': 139896043864064,
                            'frames': [{'filename': '/raid/zdevito/pytorch/torch/nn/modules/module.py',
                                        'line': 745,
                                        'name': '<lambda>'},
                                        {'filename': '/raid/zdevito/pytorch/torch/nn/modules/module.py',
                                        'line': 657,
                                        'name': '_apply'},
                                        ...],
                            'real_size': 1179648}],
                'size': 1179648,
                'state': 'active_allocated'},
                ...
                ]
    'device': 0,
    'segment_type': 'large',
    'stream': 0,
    'total_size': 20971520},
    ...]

The snapshot is a list of `Segment` dictionaries with this structure:

    from typing import TypedDict, List

    class Segment(TypedDict):
        address: int
        total_size: int #  cudaMalloc'd size of segment
        stream: int
        segment_type: str # 'large' (>1MB) or 'small'
        allocated_size: int # size of memory in use
        active_size: int # size of memory in use or in active_awaiting_free state
        blocks : List[Block]

    class Block(TypedDict):
        size: int
        state: str # 'active_allocated', used by a tensor
                   # 'active_awaiting_free', we are waiting for another stream to finish using
                   #                         this, then it will become free
                   # 'inactive', free for reuse
        history: List[History]

    class History(TypedDict):
        addr: int
        frames : List[Frame] # stack trace when address was last allocated
                             # most recent frame first
        real_size: int # unrounded size requested from the allocator

    class Frame(TypedDict):
        filename: str
        line: int
        name: str

    class Snapshot(TypedDict):
        segments : List[Segment]

    snapshot : Snapshot = torch.cuda.memory._snapshot()


`Segment`s are the memory directly requested from cudaMalloc and cached by the allocator. Because we might only be using part of one of these segments, the caching allocator splits these up into one or more `Block`s. All of a block is always in the same allocation state. With `_record_memory_history` each block will also record a `History` object that remembers the last allocation placed in that block, including its stack trace as a list of `Frame`s. For `active_allocated` blocks there will be a single history for what exists in the block and is currently allocated. For `inactive` blocks, there may be multiple entries that record the last things that lived in the memory of block. The reason there might be more than one is because the allocator will merge split blocks when they become free, and it records the history of both splits. To avoid recording a huge amount of history, we only keep history for ranges of the block that do not overlap with anything newer.

Saving snapshots
----------------

The snapshots are designed to be pickled so that they can be viewed offline later:

    from pickle import dump
    dump(snapshot, open('snapshot.pickle', 'wb'))

The file [_memory_viz.py](https://github.com/pytorch/pytorch/blob/master/torch/cuda/_memory_viz.py) can be used directly as an interactive command to work with saved snapshots:

    $ wget https://raw.githubusercontent.com/pytorch/pytorch/master/torch/cuda/_memory_viz.py
    $ python _memory_viz.py stats snapshot.pickle
    {'active_allocated': 65.9MiB,
     'inactive': 40.1MiB,
     'segments': 14,
     'total_size': 106.0MiB}

Visualizing snapshots
---------------------
The `_memory_viz.py` tool can also generate flamegraph visualizations of memory:

    $ python _memory_viz.py memory snapshot.py -o memory.svg

This produces a visualization that segements all the bytes in the allocator in different classes (click for an interactive view):

[![allocated memory](/assets/memory.svg)](/assets/memory.svg)

Flamegraphs visualizations are a way breaking down how a resource (such as memory) is used into different categories that can then be further broken down into even more finegrained categories.

The size of segments along the x-axis corresponds to the number of bytes in that segment, while the depth along the y-axis shows the different categories into which we are classifying how the memory is used. Everything in the tower above the `active_allocated` category on the left is currently allocated, while the tower above the `inactive` category holds the memory that is cached in the allocator by not in use. Above these broad categories we further categorize how the memory is used by the stack frames active when it was allocated. In this visualization we can see one stack pattern with lots of `_apply` frames that holds the model parameters (which were created in `.cuda()` which calls `_apply`). The other stack pattern in the `active_allocated` class includes `resnet.py:285:forward` and represents the activations saved for the backward pass. Another similar tower exists in the `inactive` category with the blocks that were used for temporaries in the foward pass.

The `memory` view gives a good overview of how the memory is being used. For debugging allocator issues in particular, though, it is useful to first categorized memory into individual `Segment` objects, which are the invidual `cudaMalloc` segments that allocated tracks:

    $ python _memory_viz.py segments snapshot.py -o segments.svg

[![segment memory](/assets/segments.svg)](/assets/segments.svg)

This view has a separate tower for each segment (see the `seg_<num>` categories), and lets you see how invidiual allocations get packed into the segements. The interactive view lets you zoom in on individual segments by clicking them.

Comparing snapshots
-------------------

The visualizer can also generate a visualization that shows the segments that have been added and removed between two different snapshots. For instance, we can re-run the model with a larger input, and see how the allocator requested more memory for the larger temporaries:


    input8 = torch.rand(8, 3, 224, 224, device='cuda')
    output = model(input8)
    snapshot = torch.cuda.memory._snapshot()
    dump(snapshot, open('snapshot2.pickle', 'wb'))

---

    $ python _memory_viz.py compare snapshot.pickle snapshot2.pickle  -o segments2.svg

[![segment memory](/assets/segments2.svg)](/assets/segments2.svg)

The comparison view shows just the new segments, which can help figure out what code paths prompted more memory to be allocated:

    $ python _memory_viz.py compare snapshot.pickle snapshot2.pickle  -o compare.svg
    only_before = []
    only_after = [140636932014080, 140636827156480, 140634912456704, 140634839056384, 140634843250688, 140634841153536, 140634866319360, 140634811793408, 140634845347840, $ 140636806184960, 140636778921984, 140634878902272]


[![compare memory](/assets/compare.svg)](/assets/compare.svg)

Custom analysis
---------------

If these vizualizations are not sufficient, the snapshot format, and `_memory_viz.py` are simple enough that they can be modified to produce custom views, filter out out noise like small allocations, or calculate other aggregate statistics.


Generating Snapshots when Out of Memory
---------------------------------------
When debugging how your program runs out of memory, one helpful time to generate a snapshot is during an OutOfMemory exception itself, we can do that today by registering an observer with the allocator that will be called everytime it is about to raise an OutOfMemoryError before any memory has been release while unwinding the exception:

    def oom_observer(device, alloc, device_alloc, device_free):
        # snapshot right after an OOM happened
        print('saving allocated state during OOM')
        snapshot = torch.cuda.memory._snapshot()
        dump(snapshot, open('oom_snapshot.pickle', 'wb'))

    torch._C._cuda_attach_out_of_memory_observer(oom_observer)

When running out of memory, this function may be called multiple times. This is because some ops, convolution in particular, use scratch spaces to speed up execution and can recover from an OOM error by using an implementation with a smaller footprint.


Understanding terminology
-------------------------

The visualization uses a number of terms that are useful to understand:

* `stream_<N>` - the stream that is associated with the memory. The allocator keeps a separate cache for each stream.
* `active_allocated` - memory that is in use.
* `inactive` - memory that is considered "reserved" by the allocator but is availiable for new allocations
* `active_awaiting_free` - on the CPU we have seen the last use of this memory, but we are waiting on another CUDA stream to finish using it (happens due to `t.record_stream(s)` calls).
* `<non-python>` - we didn't capture any Python stacks for when this allocation was created. Most likely this was a tensor created by our C++ autograd engine.
* `<eval ...>.forward` - These are stack frames for code generated with `torch.fx`.
* `<gaps>` - These are parts of segments that are inactive but do not have any stack of history associated with them (either they have never been allocated yet, or we removed it because it overlapped with some other piece of newer history)
