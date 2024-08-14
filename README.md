# Rock-thin-section-image-classification
Microscopic rock thin section image recognition is crucial in rock mineral analysis. Typically,53
deep learning models are used to automate expert knowledge, but the scarcity of samples in cer-54
tain categories limits the available training data, affecting the performance of traditional deep55
learning models. This paper proposes a novel few-shot learning model to address the challenge56
of classifying rock thin section images under limited sample conditions. Based on advanced57
few-shot learning processes involving pre-training and meta-training, we first introduce a Cross58
Attention Feature Fusion (CAFF) module. This module generates new features by plane polar-59
ized light images (PPL) and cross-polarized light images (XPL) of rock thin sections under a60
microscope, integrating these with the original features through autonomous learning to obtain61
more comprehensive features. Secondly, we propose a Feature Selection (FS) module based on62
the prototypical network (ProtoNet). This module extracts key feature dimensions by focusing on63
representative features within the same class and distinguishing features between classes, helping64
the model concentrate on crucial dimensions to mitigate the impact of feature sparsity. Finally,65
evaluated on the Nanjing University rock teaching sample dataset using ResNet50 and Swin-66
Transformer pre-trained on ImageNet-1000k, the ProtoNet+CAFF+FS model achieved average67
classification accuracies of 96.70% and 99.16% in the 5-Way 5-Shot few-shot task, respectively,68
surpassing traditional methods and demonstrating the effectiveness of the proposed modules.
