# Optimal Growth Schedules for Batch Size and Learning Rate in SGD that Reduce SFO Complexity
Source code for reproducing our paper's experiments.

# Abstract
The unprecedented growth of deep learning models has enabled remarkable advances but introduced substantial computational bottlenecks.
A key factor contributing to training efficiency is batch-size and learning-rate scheduling in stochastic gradient methods.
However, naive scheduling of these hyperparameters can degrade optimization efficiency and compromise generalization.
Motivated by recent theoretical insights, we investigated how the batch size and learning rate should be increased during training to balance efficiency and convergence.
We analyzed this problem on the basis of stochastic first-order oracle (SFO) complexity, defined as the expected number of gradient evaluations needed to reach an $\epsilon$–approximate stationary point of the empirical loss.
We theoretically derived optimal growth schedules for the batch size and learning rate that reduce SFO complexity and validated them through extensive experiments.
Our results offer both theoretical insights and practical guidelines for scalable and efficient large-batch training in deep learning.

# Usage

To train a model on **CIFAR-100**, run `cifar100.py` with a JSON file specifying the training parameters. Optionally, use the `--cuda_device` argument to choose a CUDA device. The default is device `0`:

```bash
python cifar100.py XXXXX.json --cuda_device 1
```

For more details about configuring checkpoints, refer to the `checkpoint_path` section in the **Parameters Description**.

### Customizing Training

To customize the training process, modify the parameters in the JSON file and rerun the script. You can adjust the model architecture, learning rate, batch size, and other parameters to explore different training schedulers and observe their effects on model performance.

## Example JSON Configuration
The following JSON configuration file is located at `src/json/const.json`:
```
{
    "model": "resnet18",
    "bs": 16,
    "bs_method": "constant",
    "lr": 0.1,
    "lr_method": "constant",
    "epochs": 200,
    "csv_path": "../result/CIFAR100/exp_comp/bs_const_lr_const/run1/"
}
```
### Parameters Description
| Parameter | Value | Description |
| :-------- | :---- | :---------- |
| `model` | `"resnet18"`, `"WideResNet28_10"`, etc. | Specifies the model architecture to use. |
| `bs_method` | `"constant"`, `"linear_growth"`, `"exp_growth"` | Method for adjusting the batch size. |
|`lr_method`| `"constant"`, `"linear_growth"`, `"exp_growth"` |Method for adjusting the learning rate.|
|`bs`|`int` (e.g., `16`)| The initial batch size for the optimizer. |
|`lr`|`float` (e.g., `0.1`)| The initial learning rate for the optimizer. |
|`epochs`|`int` (e.g., `200`)|The total number of epochs for training.|
|`incr_interval`|`int` (e.g., `20`)|Interval (in epochs) at which the batch size and learning rate will increase.|
|`coefficient`|`int` (e.g., `16`)|The factor by which the batch size increases after each interval. Used when `bs_method` is `"linear_growth"`.|
|`bs_exp_rate`|`float` (e.g., `2.0`)|The factor by which the batch size increases after each interval. Used when `bs_method` is `"exp_growth"`.|
|`lr_exp_rate`| `float` (e.g., `1.4`) |The factor by which the learning rate increases after each interval. Used when `lr_method` is `"exp_growth"`.|
|`csv_path`|`str` (e.g., `"path/to/result/csv/"`)|Specifies the directory where CSV files will be saved. Four CSV files—`train.csv`, `test.csv`, `norm.csv`, and `lr_bs.csv`—will be saved in this directory.|

