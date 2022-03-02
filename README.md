# Cross-Lingual-Adversarial-Domain-Adaptation
Pytorch Implementations for AAAI-22 Paper: Cross-Lingual Adversarial Domain Adaptation for Novice Programming



#### Model Files
For model structure implementations
1. astnn.py
2. cl_astnn.py
3. tlsm.py



#### Functional Files
For utils and processing of programming data
1. utils/
2. pipeline.py
----
Please note that pipeline.py needed to be run as the first step to get the embeddings for Java/Snap to run any experiments.



#### Trainer Files
For training and testing process
1. trainer.py
2. cl_train.py


#### Main Files
For main experiments
1. train_astnn_java.py
2. train_astnn_snap.py
3. train_clastnn.py
3. train_clastnn_temporal.py
----
Please note that 1&2 needed to be run first to generate the best models of ASTNN, which will be used to initialize CrossLing models.