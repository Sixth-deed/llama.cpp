1. 目前input tensor都是cuda host类型，可能不适合CPU直接计算，尝试修改计算图，多一份CPU buft的拷贝用于计算
2. CPU计算可能会受到并发的H2D数据传输的影响，两者争抢DMA？尝试：设置线程数、亲和性、新线程处理数据传输等方法会不会有改善
3. 修了几处bug，仍然输出不稳定，可能还存在问题
4. 进一步测试性能数据，测试在计算压力增大的情况下的性能表现
5. 核心修改（可选）：不使用层的分割方法，开启opoffload情况下计算图会被切分的十分复杂，尝试在此基础上设计并发数据传输
6. 核心修改：加入DISK，将weight进一步offload，减少CPU内存使用


我先在的想法是，不再以层为调度单位，而是以tensor为粒度。在原本的llama.cpp的系统中，tensor会被offload到GPU执行，计算图在后端切换的地方会发生切割split。所有的split串行执行，split的input也会在执行前串行拷贝。我想优化这个设计，首先split的input有些是权重或者kvcache，这些数据是没有计算的数据依赖的，也就是说他们可以提前开始传输和CPU/GPU计算并发。
第二步修改则是重新考虑哪些算子需要被offload到GPU去执行，原本的llama.cpp会根据该算子的bs或length维度来决定是否是计算量比较大的算子，这个部分是hardcoded。实际情况根据硬件和模型的不同应该是有不同的判断标准的。那我需要一套预先profile的工具，针对给定模型和硬件，测试数据传输带宽、CPU/GPU对不同大小的输入的计算效率，由这些数据来设计算法决定是否要offload该算子。这里算法的判断逻辑该怎么设计？要考虑到一个很重要的点，现在一部分数据传输（占大头的权重和cache）和计算是并发的，要考虑这个传输时间是否适合和该tensor之前的计算并发以及是否能带来该算子的计算提升。传输时间可以用带宽和数据大小提前计算，计算效率是否只能profile出来？


1. profile->tensor layout (overrides)
2. load model
3. process prompt (input)
4. pre-assign op_offload alg -> offloaded cpu weight tensors
    <!-- a. P(usually all)
    b. D(1. need offload 2. pipo method for non-offload) -->
5. create ctx 
    a. build_gf: use dynamic tensors for offload tensors in gf
    b. split: based on pre-assign / force split on dynamic tensors
6. decode

- 直接从大到小排列 tensor 然后 override 特定的 tensor 效果并不好，因为没有充分利用offload。
- 关于 kv cache. set rows 和 flash attntion 算子不涉及别的 weight，kv cache本身被硬编码到 cuda，这俩算子会被 llama.cpp 分配到 cuda 上，这与算法的预期是不符合的。算法预期两个权重都在cpu上的话，两个权重所在算子之间的所有算子都会是在 cpu 上。现在存在 wk\wq\wk 和 attn_output.weight 在 cpu 上，但中间的 flash attn 算子不在 cpu 上而是在 gpu 上计算的情况。
  - 一种修改方向是让算法加上这种情况，这样会比较臃肿，可能要引入更多的额外内存用于表达 dp 状态
  - 另一个方向是更改pipo框架，把 kv cache 也纳入相似的 override 体系中。
- prefill 的瓶颈在于 mem 传输和计算没什么并发，dynamic tensor 的传输是 per split 的，而prefill 阶段一个 dynamic tensor 就会有一个 split。落实到 prefill 上就是 compute split-> mem cpy -> compute split -> mem cpy
  - 这个改好了应该能大幅提升 prefill 速度

