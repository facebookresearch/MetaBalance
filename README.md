# MetaBalance: Improving Multi-Task Recommendations via Adapting Gradient Magnitudes of Auxiliary Tasks

### Introduction

This repo contains the pytorch implementation of MetaBalance and an example main file to call MetaBalance:<br>
> MetaBalance: Improving Multi-Task Recommendations via Adapting Gradient Magnitudes of Auxiliary Tasks<br>
> The Web Conference, 2022.<br>
> Yun He, Xue Feng, Cheng Cheng, Geng Ji, Yunsong Guo and James Caverlee.<br>
> Meta AI and Texas A&M University.<br>
> A majority of this work was done while the first author was interning at Meta AI.<br>

In many personalized recommendation scenarios, the generalization ability of a target task can be improved via learning with additional auxiliary tasks alongside this target task on a multi-task network. However, this method often suffers from a serious optimization imbalance problem. On the one hand, one or more auxiliary tasks might have a larger influence than the target task and even dominate the network weights, resulting in worse recommendation accuracy for the target task. On the other hand, the influence of one or more auxiliary tasks might be too weak to assist the target task. More challenging is that this imbalance dynamically changes throughout the training process and varies across the parts of the same network. We propose a new method: MetaBalance to balance auxiliary losses via directly manipulating their gradients w.r.t the shared parameters in the multi-task network. Specifically, in each training iteration and adaptively for each part of the network, the gradient of an auxiliary loss is carefully reduced or enlarged to have a closer magnitude to the gradient of the target loss, preventing auxiliary tasks from being so strong that dominate the target task or too weak to help the target task. Moreover, the proximity between the gradient magnitudes can be flexibly adjusted to adapt MetaBalance to different scenarios. The experiments show that our proposed method achieves a significant improvement of 8.34% in terms of NDCG@10 upon the strongest baseline on two real-world datasets.

### Acknowledgement
The technique of calculating the Moving Average of Gradient Magnitudes in this paper is learned from https://github.com/ItzikMalkiel/MTAdam. Th first author is Itzik Malkiel. Thanks to them! 

### Citation
TBD

## License

See the [LICENSE](LICENSE) file for more details.
The majority of SentAugment is licensed under CC-BY-NC. 
