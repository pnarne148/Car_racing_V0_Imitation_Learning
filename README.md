# Imitation Learning

Architecture:
1) 3 Convolutional layers to extract hierarchical features from input data.
2) 3 fully connected layers for the final classification.
3) Batch Normalization layers to accelerate the training process.
4) ReLU and sigmoid activation functions are employed for introducing non-linearity between the layers.


Used DAgger to improve my reward after regression models were applied. 




My Dagger iteration algorithm:
1) Policy Execution: The learned policy (infer_action) is executed in the environment, generating an episode of behavior.
2) Expert Correction: An expert provides corrective actions for “each state” encountered during policy execution. All the states in the episode are saved. Expert data is collected using a timer.
3) Data Aggregation: The new observations and expert actions are added to the training data.
4) Policy Refinement: The policy is re-trained using the aggregated dataset.
5) Model Saving: The updated model is saved after each iteration.
Dagger implementation can be found in dagger.py file.

