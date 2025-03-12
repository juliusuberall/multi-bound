import argparse
import yaml
import jax
import jax.numpy as jnp
import numpy as np
import optax
from utils.sampler import RGBAImageSampler
from utils.model import *

if __name__ == "__main__":
    
    # Initialize random key
    key = jax.random.PRNGKey(28)

    # Load YAML config by parsing invoking arguments
    # to get access to model architecture definitions
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the model config file")
    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # Load and set up training signal
    sampler = RGBAImageSampler("data/img/dolphin_color.png")
    sampler.check_signal(key)

    # Create and train model
    for c in config:

        ## Retrieve model class type
        model_type = model_registry[c]

        ## Register class as jit compilable
        jax.tree_util.register_pytree_node(
            model_type,
            model_type.flatten_func,
            model_type.unflatten_func
        )

        ## Create inital parameters
        model = model_type(config[c], sampler, key)

        ## Training 
        opt = optax.adam(learning_rate=0.01)
        opt_state = opt.init(model.params)
        num_epochs = 2000
        batch_size = 1024

        @jax.jit
        def update(p, opt_state, x, y):
            grads = jax.grad(model_type.loss)(p, x, y)
            updates, opt_state = opt.update(grads, opt_state)
            p = optax.apply_updates(p, updates)
            return p, opt_state

        # Training loop
        for epoch in range(num_epochs):
            x, y = sampler.sample(batch_size, key)
            model.params, opt_state = update(model.params, opt_state, x, y)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {model_type.loss(model.params, x, y)}")

    # Train model
    print("Script finished!")