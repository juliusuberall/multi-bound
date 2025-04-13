from utils.model.MoE import *
import inspect
import shutil

class MoEH(MoE):
    """
    <summary>
        Soft trained and hard iferred Mixture of Experts (MoE) implementation with gate network and n expert networks.
        Builds on top of trained soft MoE and is not intended for training due to hard forward function.
    </summary>
    """

    def __init__(self, config, sampler:DataSampler, key):
        # Delegate to parent constructor 
        super().__init__(config, sampler, key)
    
    @staticmethod
    @jax.jit
    def forward(p:MoEParams, x):
        """
        Hard forward through entire Mixture of Experts (MoE). 
        This will select the top k experts and use them for inferrence instead of using all for all.

        Args
        ----------
        p :
            Network parameters for gate and experts
        x :
            The input data to forward through the model. Expects 2D array [batch dimension, input dim]
        """

        ##### HARD CODED FOR NOW
        ##### could not find good solution in implementation
        top_k = 1

        # Get top K indicies
        ## gate output shape : [batchsize, number of experts
        gate = jax.vmap(lambda x: MoE.forward_gate(p.gate, x))(x)
        _ , idx = jax.lax.top_k(gate, top_k)


        # Define expert branches to switch to
        ## Tried a lot around, could not get anything else to work
        branches = [
            lambda x: MoE.forward_expert(p.expert1, x),
            lambda x: MoE.forward_expert(p.expert2, x)
        ]
        
        @staticmethod
        @jax.jit
        def expert_branching(x, i):
            return jax.lax.switch(i, branches, x)

        expert_out = jax.vmap(expert_branching)(x, idx.squeeze(-1))

        return expert_out
    
    @staticmethod
    @jax.jit
    def loss(p:MoEParams, x, y):
        preds = MoEH.forward(p, x)
        return jnp.mean((preds - y) ** 2)
    
    @staticmethod
    def analysis_prep():
        """
        Duplicates all soft trained moe json and declares them to be inferred hard as MoEH
        """
        folder = dir_registry['model_params_dir']
        for f in sorted(os.listdir(folder)):
            if f.split('_')[0] == 'moe':
                shutil.copy(folder + "/" + f, folder + "/" + f.split('_')[0] + "H_"+ f.split('_')[1])

    @staticmethod
    def analysis_cleanup():
        """
        Removes moeH json, since they are only needed for analysis, but are essentially the same as MoE
        """
        folder = dir_registry['model_params_dir']
        for f in sorted(os.listdir(folder)):
            if f.split('_')[0] == 'moeH':
                if os.path.exists(folder + "/" + f):
                    os.remove(folder + "/" + f)