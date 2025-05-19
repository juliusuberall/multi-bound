# Global pipeline variables

# Define expert evaluation branches dynamically 
## Uses lambda to define local, unnamed python functions which represent 
## the execution of the correct expert branch for an input
global_MoE_branches = None