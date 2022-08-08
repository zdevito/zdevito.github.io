---
layout: post
title:  "Extracting information about CUDA crashes from core dumps"
date:   2022-07-27 21:55:37 -0700
categories: ""
---

If a GPU reads invalid memory, the CUDA API will start returning `cudaErrorIllegalAddress` from all API calls:

> The device encountered a load or store instruction on an invalid memory address. This leaves the process in an inconsistent state and any further CUDA work will return the same error. To continue using CUDA, the process must be terminated and relaunched.

Because CUDA kernels get launched asynchronously from the CPU, this error will not be reported where the faulting kernel is launched, but instead at any CUDA API call after the exception actually happens on the GPU and propagates to the CPU. The `CUDA_LAUNCH_BLOCKING=1` environment variable will cause CUDA to wait for kernels to finish before returning from their launch, but this will make your program significantly slower and possibly change the timing enough that a nondeterministic issue is no longer triggered. Furthermore, if there are multiple threads using the CUDA API, the cudaErrorIllegalAddress might get reported first on one of those threads and not the launcher thread anyway. So I do not trust the stack traces even with `CUDA_LAUNCH_BLOCKING=1`.

Instead, we want to extract much more accurate information about what caused the illegal address. Like any processor, when a fault happens the SM on a GPU records information about the faulting instruction. Unfortunately, there is no in-process way I am aware of to get this information. It is accessible by attaching `cuda-gdb` or `cuda-memcheck` to the process before running it. But for rarely occurring bugs, it is not practical to re-run the process in this mode to reproduce it.

Instead, by setting the environment variable `CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1`, we can instruct CUDA to produce a core dump of the GPU's state after an exception occurs, and use cuda-gdb to examine that file later.

This post goes over how to generate an extract information from these core dumps. Even without debug information, a surprising amount of information, including the values of parameters and the instruction that faulted can be recovered.

Generate the core dump
=====================

Set `CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1` on the process that has the fault. When the fault occurs it will produce a core dump file `cudacoredump.hostname.pid`.

Use cuda-gdb to open the core dump
==================================

    $ /usr/local/cuda/bin/cuda-gdb
    (cuda-gdb) target cudacore /tmp/cudacoredump.hostname.pid
    Opening GPU coredump: /tmp/cudacoredump.hostname.pid

It should report some information about where the fault happened:

    CUDA Exception: Warp Illegal Address
    The exception was triggered at PC 0x7ff8b63ce440
    [Current focus set to CUDA kernel 0, grid 132575338, block (1240,0,0), thread (0,1,0), device 0, sm 1, warp 62, lane 0]
    #0  0x00007ff8b63ce570 in void (anonymous namespace)::softmax_warp_forward<c10::Half, c10::Half, float, 8, false, true>(c10::Half*, c10::Half const*, int, int, int, bool const*, int, bool)<<<(1824,1,1),(32,4,1)>>> ()

Relevant information here:

* The address of the instruction that triggered the Warp Illegal Address: `The exception was triggered at PC 0x7ff8b63ce440`

* The name of the kernel that was running: `softmax_warp_forward`

* The address at which execution halted: `0x00007ff8b63ce570`

Notice that the address where the GPU stops (`...570`) is _after_ the triggering address (`...440`). Memory reads happen asynchronously so the GPU will have kept execution instructions and only later finds out about the fault. Keep this in mind when looking at the value of registers because it will reflect the state where execution halted and the values of the registers used in the faulting instruction may have been overwritten.

Finally, unless debug information was included in your build, you will not see line or filename information for your code. Later steps show how a lot of information can still be recovered from the dump even without this information.

Disassemble the kernel
========================

Use `disas` to see a listing of shader assembly (SASS) for the kernel:

    (cuda-gdb) disas
    ...
    0x00007ff8b63ce420 <+1056>:  IADD3 R8, R6.reuse, 0xc0, RZ
    0x00007ff8b63ce430 <+1072>:  IADD3 R18, R6, 0xe0, RZ
    0x00007ff8b63ce440 <+1088>:  LDG.E.U8.SYS R19, [R2+0xe0]
    0x00007ff8b63ce450 <+1104>:  ISETP.GE.AND P3, PT, R8, R13, PT
    ...

To see the faulting instruction find the PC that matches it:

    0x00007ff8b63ce440 <+1088>:  LDG.E.U8.SYS R19, [R2+0xe0]

In this case it is LDG "load from global memory" and it is reading 1 byte ("U8") from the address [R2+0xe0] into the register R19. Presumably R2+0xe0 is out of bounds.

Examine registers
=================

Use `info reg` to see that values of all GPU registers:

    (cuda-gdb) info reg
    R0             0xb8198             754072
    R1             0xfffc80            16776320
    R2             0xff800000          -8388608
    R3             0xff800000          -8388608
    R4             0xff800000          -8388608
    R5             0x7ff8              32760
    R6             0x0                 0
    R7             0x2                 2
    R8             0x407ce000          1081925632
    ...

Even though we have a value for R2 here, it turns out that R2 was overwritten between PC ...440 and ...570, so we can't easily find the value of the faulting address.

Read GPU memory
===============

Use print to read values from memory:

    # read a void* from CUDA's global memory:
    (cuda-gdb) print *(void * @global *)0x7ff841000000

    # read an int from CUDA's global memory
    (cuda-gdb) print *(int @global *)0x7ff841000000

Recover the parameters passed to the kernel
===========================================

The parameters to a kernel are passed in constant "parameter" memory. The instructions that load them include references to constant memory like `c[0x0][0x174]`:

    0x00007ff8b63ce080 <+128>:   IMAD R0, R3.reuse, c[0x0][0x174], R6

You can read this memory using:

    (cuda-gdb) print *(int @parameter *)0x174
    152


To actually get the values of all the kernel parameters, we need to understand how they are arranged in this memory. Say the kernel has arguments:

    _global__ void softmax_warp_forward(
      output_t *dst,
      const input_t *src,
      int batch_size, int stride,
      int element_count,
      const bool *mask = nullptr,
      const int head_chunk_size = -1, bool is_transformer_mask = false) {
    ...

The layout of the arguments in constant memory is the same as if they were put into a struct:

    struct Args {                  // offset
        output_t *dst;             // 0
        const input_t *src;        // 8
        int batch_size;            // 16
        int stride;                // 20
        int element_count;         // 24
        // <4 bytes padding>
        const bool *mask;          // 32
        const int head_chunk_size; // 40
        bool is_transformer_mask;  // 44
    };

This generally means that values are aligned to the next multiple of their size (8 byte types are aligned to 8 byte multiples), inserting some padding bytes as necessary.

The start of the kernel parameters is not at 0x0 (lower addresses contain some additional meta-data about the kernel), and you might have to look at all the references to `c[0x0][...]` in the assembly to see where the parameter buffer likely starts based on how the values are used. In my run it looks like the parameters start at 0x160, because this was the smallest reference to constant memory that cuda-gdb would return a value for.

Once you know the layout and the start address, `print` can be used to get the values (specify the right type in the print):

    # stride
    (cuda-gdb) print *(int @parameter *) (0x160 + 20)
    152

More Info
=========

[SASS Documentation](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html) has some more documentation of the assembly language being run, but it is not fully documented and changes between GPU generations.

