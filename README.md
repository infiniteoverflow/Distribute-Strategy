# Distribute-Strategy

One very useful addition to TensorFlow 2.x is the possibility to train models using distributed GPUs, multiple machines, and TPUs in a very simple way with very few additional lines of code. `tf.distribute.Strategy` is the TensorFlow API used in this case and it supports both `tf.keras` and `tf.estimator` APIs and eager execution. You can switch between GPUs, TPUs, and multiple machines by just changing the strategy instance. Strategies can be synchronous, where all workers train over different slices of input data in a form of sync data parallel computation, or asynchronous, where updates from the optimizers are not happening in sync. All that strategies require is that the data should be loaded in batches using the `tf.data.Dataset` API.

Source : **Deep Learning with TensorFlow 2 and Keras : The Book**

If we want to have synchronous distributed training on multiple GPUs on one machine, there are two things that we need to do: (1) We need to load the data in a way that will be distributed into the GPUs, and (2) We need to distribute some computations into the GPUs too:

1. In order to load our data in a way that can be distributed into the GPUs, we simply need a `tf.data.Dataset` (which has already been discussed in the previous paragraphs). If we do not have a `tf.data.Dataset` but we have a normal tensor, then we can easily convert the latter into the former using `tf.data.Dataset.from_tensors_slices()`. This will take a tensor in memory and return a source dataset, the elements of which are slices of the given tensor.

2. In order to distribute some computations to GPUs, we instantiate a `distribution = tf.distribute.MirroredStrategy()` object, which supports synchronous distributed training on multiple GPUs on one machine. Then, we move the creation and compilation of the Keras model inside the `strategy.scope()`. Note that each variable in the model is mirrored across all the replicas.

For instance, if using **MirroredStrategy()** with two GPUs, each batch of size 256 will be divided among the two GPUs, with each of them receiving 128 input examples for each step
