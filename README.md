# **Federated Learning on Heterogeneous and Long-Tailed Data via Classifier Re-Training with Federated Features**

This is an official implementation of the following paper:

> Xinyi Shang, **Yang Lu***, Gang Huang, and Hanzi Wang.
> **Federated Learning on Heterogeneous and Long-Tailed Data via Classifier Re-Training with Federated Features**
> *International Joint Conference on Artificial Intelligence (IJCAI), 2022* 



**Abstract:** Federated learning (FL) provides a privacy-preserving solution for distributed machine learning tasks. One challenging problem that severely damages the performance of FL models is the co-occurrence of data heterogeneity and long-tail distribution, which frequently appears in real FL applications. In this paper, we first reveal an intriguing fact that the biased classifier is the primary factor leading to the poor performance of the global model. Motivated by the above finding, we propose a novel and privacy-preserving FL method for heterogeneous and long-tailed data via Classifier Re-training with Federated Features (CReFF).  The classifier re-trained on federated features can produce comparable performance as the one re-trained on real data in a privacy-preserving manner without information leakage of local data or class distribution. Experiments on several benchmark datasets show that the proposed CReFF is an effective solution to obtain a promising FL model under heterogeneous and long-tailed data. Comparative results with the state-of-the-art FL methods also validate the superiority of CReFF.



### Dependencies

- python 3.7.9 (Anaconda)
- PyTorch 1.7.0
- torchvision 0.8.1
- CUDA 11.2
- cuDNN 8.0.4



### Dataset

- CIFAR-10
- CIFAR-100
- ImageNet-LT



### Parameters

The following arguments to the `./options.py` file control the important parameters of the experiment.

| Argument                    | Description                                       |
| --------------------------- | ------------------------------------------------- |
| `num_classes`               | Number of classes                                 |
| `num_clients`               | Number of all clients.                            |
| `num_online_clients`        | Number of participating local clients.            |
| `num_rounds`                | Number of communication rounds.                   |
| `num_epochs_local_training` | Number of local epochs.                           |
| `batch_size_local_training` | Batch size of local training.                     |
| `match_epoch`               | Number of optimizing federated features.          |
| `crt_epoch`                 | Number of re-training classifier.                 |
| `ipc`                       | Number of federated features per class.           |
| `lr_local_training`         | Learning rate of client updating.                 |
| `lr_feature`                | Learning rate of federated features optimization. |
| `lr_net`                    | Learning rate of classifier re-training           |
| `non_iid_alpha`             | Control the degree of heterogeneity.              |
| `imb_factor`                | Control the degree of imbalance.                  |



### Usage

Here is an example to run CReFF on CIFAR-10 with imb_factor=0.01:

```python
python main.py --num_classrs=10 \ 
--num_clients=20 \
--num_online_clients=8 \
--num_rounds=200 \
--num_epochs_local_training=10 \
--batch_size_local_training=32 \
--match_epoch=100 \
--ctr_epoch=300 \
--ipc=100 \
--lr_local_training=0.1 \
--lr_feature=0.1 \
--lr_net=0.01 \
--non-iid_alpha=0.5 \
--imb_factor=0.01 \ 
```



