---
layout: post
title:  "Vizualizing PyTorch memory usage over time"
date:   2022-12-08 21:55:37 -0700
categories: ""
---

Vizualizing PyTorch memory usage over time
==========================================
As described in a [previous post](https://zdevito.github.io/2022/08/16/memory-snapshots.html), _memory snapshots_ are a way to dump and visualize all the state about how CUDA memory in PyTorch is allocated. This information is especially useful when trying to debug why a program is running out of memory (an OOM) because it lets you see the stack traces for all allocated memory, and how it fits in to the memory caches that our [caching allocator](https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html) uses.

However, sometimes it is useful to see the series of allocation _events_ that led up to running out of memory in addition to the state of memory when the program crashed. _Memory traces_ provide this facility by supplementing the snapshot information with trace events related to memory.

Generating Memory Snapshots with Traces
---------------------------------------

Like snapshots, we have to enable memory recording:

    torch.cuda.memory._record_memory_history(True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,

            # record stack information for the trace events
            trace_alloc_record_context=True)

To limit the total size of the trace event buffer, we limit it to the last `trace_alloc_max_entries` before a snapshot is taken, but the facility can easily record hundreds of thousands of events. Recording these traces is pretty fast (~1us per allocation, a normal PyTorch kernel call takes at least 8 us), and adds almost no extra time if the program was already recording for memory snapshots.  Taking a snapshot is like before:

    from torchvision.models import resnet18
    from pprint import pprint

    model = resnet18().cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    input = torch.rand(8, 3, 224, 224, device='cuda')
    labels = torch.zeros(8, dtype=torch.long, device='cuda')

    model.train()

    outputs = model(input)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    snapshot = torch.cuda.memory._snapshot()
    pprint(snapshot['device_traces'][0])

There is one trace per device, and each trace looks like:

    [{'action': 'segment_alloc',
    'addr': 139934079909888,
    'frames': [{'filename': '/raid/zdevito/pytorch/torch/nn/modules/module.py',
                'line': 890,
                'name': '<lambda>'},
                {'filename': '/raid/zdevito/pytorch/torch/nn/modules/module.py',
                'line': 805,
                'name': '_apply'},
                ...],
    'size': 2097152,
    'stream': 0},
    {'action': 'alloc',
    'addr': 139934079909888,
    'frames': [{'filename': '/raid/zdevito/pytorch/torch/nn/modules/module.py',
                'line': 890,
                'name': '<lambda>'},
                {'filename': '/raid/zdevito/pytorch/torch/nn/modules/module.py',
                'line': 805,
                'name': '_apply'},
                ...],
    'size': 37632,
    'stream': 0},
    ...]

Device traces are a list of `TraceEntry` dictionaries with this structure:

    from typing import TypedDict, List

    class TraceEntry(TypedDict):
        action: str # one of
        #'alloc', memory allocated
        #'free_requested', the allocated received a call to free memory
        #'free_completed', the memory that was requested to be freed is now
        #                   able to be used in future allocation calls
        #'segment_alloc', the caching allocator ask cudaMalloc for more memory
        #                 and added it as a segment in its cache
        #'segment_free', the caching allocator called cudaFree to return memory
        #                to cuda possibly trying free up memory to
        #                allocate more segments or because empty_caches was called
        #'oom',          the allocator threw an OOM exception. 'size' is
        #                the requested number of bytes that did not succeed
        #'snapshot'      the allocator generated a memory snapshot
        #                useful to coorelate a previously taken
        #                snapshot with this trace

        addr: int # not present for OOM
        frames: List[Frame]
        size: int
        stream: int
        device_free: int # only present for OOM, the amount of
                         # memory cuda still reports to be free


    class Frame(TypedDict):
        filename: str
        line: int
        name: str


    traces: List[List[TraceEntry]] = torch.cuda.memory._snapshot()['device_traces']
    # one list per each device
    trace_device_3: List[TraceEntry] = traces[3]

Saving snapshots
----------------
Since the traces are just part of the snapshot, they can be pickled in the same way to view offline later.

    from pickle import dump
    with open('snapshot.pickle', 'wb') as f:
        dump(snapshot, f)

The file [_memory_viz.py](https://github.com/pytorch/pytorch/blob/master/torch/cuda/_memory_viz.py) can be downloaded and run independently of pytorch to view the traces textually:

    $ wget https://raw.githubusercontent.com/pytorch/pytorch/master/torch/cuda/_memory_viz.py
    $ python _memory_viz.py stats snapshot.pickle
    Device 0 ----------------
    291 entries
    a = cudaMalloc(139838718214144, 2.0MiB)
    b = a[0:36.8KiB]
    c = a[37888:256.0B]
    d = a[38400:256.0B]
    e = a[38912:256.0B]
    f = a[39424:256.0B]
    g = a[39936:8.0B]
    ...

Visualizing traces
------------------
The `_memory_viz.py` can generate interactive html plots that let you explore the allocated memory as it existed over time:

    $ python _memory_viz.py trace_plot snapshot.pickle -o trace.html

The visualization plots the total amount of memory allocated on the Y axis, with memory events over time on the X axis. [Click for an interactive view](/assets/trace.html):

![trace](/assets/trace.png)

Brushing over individual allocations provides the stack trace where they were allocated.

Looking at the visualization makes it easy to see patterns in memory usage during training. The forward pass is clear as the memory for activtions accumulates waiting for its use in the backward pass. Then as the backward pass starts, those saved activations start to be freed step-by-step as gradient tensors get allocated. The leads to the common pattern where max memory usage occurs somewhere early in the backward pass. Then at the end, the optimizer step, which allocates its own temporaries is visible.

Another interesting pattern is the spikes in usage that occur both in forward and backward. Looking at the forward pass ones with stack information reveals they are part of the `_conv_forward` operator and are likely the temporary buffers allocated to perform the fastest convolution type in cudnn. One common pattern to see when near maximum memory usage is for cudnn to run out of memory and try a different algorithm and succeed. This visualization makes it clear this happens because these spikes will hit the memory max first.


The chart also allows panning and zooming using the minimap or dragging the chart around, where we can zoom into one of these spikes in memory usage in the backward pass:

![trace2](/assets/trace2.png)


Generating Traces when Out of Memory
---------------------------------------
With memory tracing turned on, it can be helpful to generate a snapshot and a trace righ tat the point of running out of memory by registering an observer with the allocator that will be called everytime it is about to raise an OutOfMemoryError:

    def oom_observer(device, alloc, device_alloc, device_free):
        # snapshot right after an OOM happened
        print('saving allocated state during OOM')
        snapshot = torch.cuda.memory._snapshot()
        dump(snapshot, open('oom_snapshot.pickle', 'wb'))

    torch._C._cuda_attach_out_of_memory_observer(oom_observer)

When running out of memory, this function may be called multiple times because as we saw with the spikes earlier convolution might run out of memory and retry with an algorithm that uses less scratch space.
