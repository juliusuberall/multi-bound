from utils.model.BaseModel import BaseModel
from utils.sampler import *
from utils.parameter.MoEParams import *

class MoE(BaseModel):
    """
    <summary>
        Mixture of Experts (MoE) implementation with gate network and n expert networks.
    </summary>
    """

    def __init__(self, config, sampler:DataSampler, key):
        """
        Instantiates Mixture of Experts for the given sampler.

        Args
        ----------
        config :
            The model configuration.
        sampler :
            The data sampler used for training the model.
        key :
            Jax random key used for sampling.
        """
        self.learning_rate = config["learning_rate"]
        
        # Assign correct MoE parameter struct based on number of experts
        p = moe_params_registry[config["n_experts"]]()
        for n in p.__dataclass_fields__:
            
            ## Init network layer correct depending on gate or expert
            layer = []
            if n == "gate" :
                layer = self.init_layer(
                    [sampler.x_dim] + config["gate_hidden_layer"] + [config["n_experts"]],
                    key
                )
            else :
                layer = self.init_layer(
                    [sampler.x_dim] + config["expert_hidden_layer"] + [sampler.y_dim],
                    key
                )

            ## **{x:y} transforms into keyword argument x=y
            p = p.replace(**{n: layer})
        self.params = p
    
    @staticmethod
    @jax.jit
    def forward_expert(p:MoEParams, x):
        """
        Forward through expert network.

        Args
        ----------
        p :
            Expert network parameters
        x :
            The input data to forward through the model
        """
        for W, b in p:
            x = jax.nn.leaky_relu(jnp.dot(x, W) + b, 0.01)
        return x

    @staticmethod
    @jax.jit
    def forward_gate(p:MoEParams, x):
        """
        Forward through gate network and pick top K.

        Args
        ----------
        p :
            Gate network parameters
        x :
            The input data to forward through the model
        """
        for W, b in p[:-1]:
            x = jax.nn.leaky_relu(jnp.dot(x, W) + b, 0.01)
        final_w, final_b = p[-1]
        x = jax.nn.softmax(jax.nn.sigmoid(jnp.dot(x, final_w) + final_b))
        return x

    @staticmethod
    @jax.jit
    def forward(p:MoEParams, x):
        """
        Forward through entire Mixture of Experts (MoE).

        Args
        ----------
        p :
            Network parameters for gate and experts
        x :
            The input data to forward through the model
        """
        # Gating
        ## gate output shape : [batchsize, number of experts]
        gate = jax.vmap(lambda x: MoE.forward_gate(p.gate, x))(x)

        # Experting
        ## expert output shape: [sample, RGB, num_experts]
        expert_out = []
        for n in p.__dataclass_fields__:
            if n == "gate" : continue
            expert_out.append(jax.vmap(lambda x: MoE.forward_expert(p.__getattribute__(n), x))(x))
        expert_out = jnp.stack(expert_out, axis=-1)

        # Weighted sum for final prediciton
        ## final output shape: [sample, RGB, num_experts]
        final_pred = jnp.sum(jnp.tile(gate[:,None,:], (1,3,1)) * expert_out, axis=-1)

        return final_pred

    @staticmethod
    @jax.jit
    def loss(p:MoEParams, x, y):
        preds = MoE.forward(p, x)
        return jnp.mean((preds - y) ** 2)

    def full_signal_inference_IMG(): pass

    def flatten_func(): pass

    def unflatten_func(): pass

    def serialize(): pass

    def deserialize(): pass