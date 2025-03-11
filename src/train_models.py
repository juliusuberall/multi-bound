import argparse
import yaml
import jax
import jax.numpy as jnp
from utils.sampler import RGBAImageSampler

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

    # Train model
    print("Script finished!")