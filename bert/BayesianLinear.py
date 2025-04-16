import tensorflow as tf
import tensorflow_probability as tfp

#贝叶斯全连接层
class BayesianLinearLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, w_prior, b_prior, **kwargs):
        super(BayesianLinearLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.w_prior = w_prior
        self.b_prior = b_prior

    def build(self, input_shape):
        # 后验分布的参数
        self.w_posterior = tfp.layers.DenseReparameterization(
            units=self.output_dim,
            activation=None,
            # 注意这里没有 kernel_prior 和 bias_prior 参数
            # 而是直接在创建层时指定了先验分布
            kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p),
            bias_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p),
            # 其他必要的参数...
        )
        self.built = True

    def call(self, inputs, **kwargs):
        return self.w_posterior(inputs)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], self.output_dim])