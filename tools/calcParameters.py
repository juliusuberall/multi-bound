# Count single networks parameter
def count_parameter(layer):
    p = 0
    for i in range(1, len(layer)):
        p += layer[i] * layer[i-1] + layer[i]
    return p

# MLP
mlp_hidden = [3, 69, 69, 69, 3] # Considers input and output dimension to be 1
print(f"-> MLP has a total of {count_parameter(mlp_hidden)} parameters.")

# MoE
# Define MoE
n_experts = 4
expert_hidden_layer = [3, 32, 32, 32, 3]
gate_hidden_layer = [3, 16, 16, 16, n_experts]

# Calculate
p = count_parameter(expert_hidden_layer) * n_experts + count_parameter(gate_hidden_layer)
print(f"-> MoE has a total of {p} parameters.")

