import argparse
import yaml
import jax
import optax
from utils.sampler import RGBAImageSampler
from utils.model.BaseModel import *
from utils.model.registry import *

if __name__ == "__main__":
    
    # Initialize random key
    key = jax.random.PRNGKey(28)

    # Parse invoking arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the model config file")
    parser.add_argument("--data", required=True, help="Path to data object to fit")
    args = vars(parser.parse_args())

    # Load YAML config by parsing invoking arguments
    # to get access to model architecture definitions
    with open(args['config'], "r") as file:
        config = yaml.safe_load(file)

    # Load and set up training signal
    sampler = RGBAImageSampler(args['data'])
    sampler.check_signal(key)

    # Create and train model
    for c in config:
        print(f'Training "{c}"')

        ## Retrieve model class type
        model_type = model_registry[c.split('_')[0]]

        ## Create inital parameters
        model = model_type(config[c], sampler, key)

        ## Training Hyperparameters and initalization
        opt = optax.adam(learning_rate=0.01)
        opt_state = opt.init(model.params)
        epsilon = 1e-5
        epoch = 0
        batch_size = 1024

        ## Initialize losses for trainig stop when model satuared and plateaued
        x, y = sampler.sample(batch_size, key)
        previous_loss, current_loss = model_type.loss(model.params, x, y), 0

        ## Update function needs to be here because of direct
        ## referencing of opt and opt.state for JIT compilation
        @jax.jit
        def update(p, opt_state, x, y):
            grads = jax.grad(model_type.loss)(p, x, y)
            updates, opt_state = opt.update(grads, opt_state)
            p = optax.apply_updates(p, updates)
            return p, opt_state

        # Training loop
        while jnp.abs(previous_loss - current_loss) > epsilon :
            x, y = sampler.sample(batch_size, key)
            model.params, opt_state = update(model.params, opt_state, x, y)
            if epoch % 100 == 0:
                current_loss, previous_loss = model_type.loss(model.params, x, y), current_loss
                print(f"Epoch {epoch}, Loss: {current_loss}")
            epoch += 1

        # Save model parameters as JSON
        model.params.serialize(c)

        # Log Finished Training
        print(f'"{c}" trained and saved!')