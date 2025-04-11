import argparse
import yaml
import jax
import optax
import numpy as np
from collections import deque
from utils.DataSampler import RGBAImageSampler
from utils.model.BaseModel import *
from utils.model.registry import *
from utils.Analyzer import Analyzer

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
        opt = optax.adam(model.learning_rate)
        opt_state = opt.init(model.params)
        epoch = 0

        analyzer = Analyzer(sampler)

        ## Initialize losses for trainig stop when model satuared and plateaued
        x, y = sampler.sample(train_set["batch_size"], key)
        avg_previous_val_loss, val_loss = analyzer.eval_accuracy_IMG(model_type, model.params), 0
        
        ### To check if the model converged the last two epoch checkpoint
        ### losses are averaged and the difference to the current loss is 
        ### checked against a threshold 
        loss_buffer = deque(maxlen= train_set["plateau_depth"])
        loss_buffer.append(float(avg_previous_val_loss))

        ## Update function needs to be here because of direct
        ## referencing of opt and opt.state for JIT compilation
        @jax.jit
        def update(p, opt_state, x, y):
            grads = jax.grad(model_type.loss)(p, x, y)
            updates, opt_state = opt.update(grads, opt_state)
            p = optax.apply_updates(p, updates)
            return p, opt_state

        # Training loop
        while jnp.abs(avg_previous_val_loss - val_loss) > train_set["epsilon"] :

            # Split JAX random key to ensure new data sampling
            key, subkey = jax.random.split(key)

            x, y = sampler.sample(train_set["batch_size"], subkey)
            model.params, opt_state = update(model.params, opt_state, x, y)
            if epoch % 100 == 0:

                ## Validation Loss logging to detect convergence plateau
                ## if maintained over epoch checkpoints
                val_loss, previous_val_loss = analyzer.eval_accuracy_IMG(model_type, model.params), val_loss
                loss_buffer.append(float(previous_val_loss))
                avg_previous_val_loss = np.mean(loss_buffer)

                print(f"Epoch {epoch}, Val-Loss: {val_loss}")
            epoch += 1

        # Save model parameters as JSON
        model.params.serialize(c)

        # Log Finished Training
        print(f'"{c}" trained and saved!')