# HYCSE_LogisticRegression_vs_DNN
# 0x01. preprocess 
### 1-1. Transformations
##### What is [[pytorch transformations]]

### 1-3. Dataset Loading
### 1-4. Train/Val Split
### 1-5. Dataloaders
### 1-6. [[data augmentation]]를 해야하는가?
##### 1) [[Logistic Regression]]은 필요없음
- Logistic Regression sees _each rotated or flipped sample as a completely new sample_. It **can’t** learn invariance, so adding augmented data just adds noise.
##### 2) DNN에는 도움이 됨 
- [[regularization]]의 일종이므로.
##### 3) 따라서 bare한 처리
---



# 0x02. LR model construction 
### 2-1. 구조:
##### **입력 레이어 (Input Layer):**
- **Preprocessing:** 입력 이미지는 원래 형태인 `[batch_size, channels, height, width]`에서 각 샘플마다 1차원 벡터로 변환함. 예를 들어, 전체 해상도 `128 × 128`을 사용할 경우 벡터의 차원은:
	- `input_dim = height * width = 128 * 128 = 16384`임.
- **설명:** 이 Flattening 과정은 공간적 특징을 하나의 벡터로 결합하여 단순한 Linear Layer에서 처리할 수 있도록 함.

##### **출력 레이어 (Output Layer):**
- **Linear Layer:** 모델은 단일 `nn.Linear(input_dim, 1)` 레이어로 구성됨.
- **레이어 구성:** 이 Linear Layer는 Flatten된 입력 벡터에 가중치와 편향을 적용하여 각 예제당 하나의 출력 logit을 생성함.
- **BCEWithLogitsLoss의 사용:** 출력된 logit은 직접적으로 `BCEWithLogitsLoss`에 전달됨. 이 손실 함수는 내부적으로 Sigmoid activation과 binary cross-entropy 계산을 결합하여 처리함. 이로 인해 forward pass에서 명시적인 Sigmoid 함수 적용이 필요 없음.

### 2-2. Activation 및 Regularization:
##### **Activation:**
- **implicit Sigmoid Activation:** forward pass에서는 별도의 activation 함수를 사용하지 않음. 대신 `BCEWithLogitsLoss`가 내부적으로 Sigmoid를 적용하여 non-linearity를 제공함. 이는 binary classification 작업에서 최신(state-of-the-art) 방식으로 채택됨.
  
##### **Hidden Units:**
- **Hidden Layer 없음:** 모델은 Logistic Regression과 유사하게 입력 변환과 단일 Linear Layer로 구성되어 있음. Hidden Layer가 존재하지 않아 구조가 단순하며 해석하기 쉬움.

### 2-3. 구현 세부 사항:
- **Flattening 연산:** `x.view(x.size(0), -1)` 연산을 통해 입력 텐서를 `[batch_size, channels, height, width]`에서 `[batch_size, input_dim]`으로 reshape함. 이 연산은 효율적이며, 분류에 필요한 특징들을 보존함.
- **학습되는 파라미터:** 단일 Linear Transformation만 존재하므로 학습되는 파라미터는 weight matrix와 bias뿐임. 이러한 단순한 구조는 데이터가 제한된 상황이나 모델 해석이 중요한 경우에 유리함.
- **최적화 및 Regularization:** 학습 스크립트에서는 Adam optimizer(learning rate: 1e-3)와 L2 Regularization(weight_decay)를 사용함. 이는 최신 딥러닝 모델에서 일반적으로 사용되는 기법임.

---
# 0x03. DNN(3 layer) model construction

### 3-1. MODEL - Initialization (init method)
##### 1) Layer Definition for network architecture
- \# of layers $\rightarrow$ 3
- neuron # per layer 
	- $\rightarrow$ list로 받아서 [[hyper-parameter]]조정
- [[layer type]] 
	- $\rightarrow$ 배운 linear 로 사용
##### 2) [[parameter initialization]]
- $W \text{ and } b$ for each layer
	- [[He initialization]]을 사용해야 함. ($\because$ relu 썼어서)
##### 3) [[hyper-parameter]]
- learning rate $\alpha$, momentum for momentum-based optimizer, [[regularization]] coefficients, dropout rates
###### optuna 사용으로 best parameter를 뽑음
- best_params = {  
    'hidden_dim1': 939,  
    'hidden_dim2': 799,  
    'dropout_rate': 0.2963617429089822,  
    'learning_rate': 0.002844883832313113,  
    'weight_decay': 1.2973725923247895e-05  
}
##### 4) BN ([[batch normalization]])
> Optionally, include batch normalization layers to accelerate training and improve stability.

$\rightarrow$ 사용

```python
class DNN(nn.Module):  
    def __init__(self, layer_dims, learning_rate=0.001, momentum=0.9, reg_coeff=0.01, dropout_rate=0.1):  
        super(DNN, self).__init__()  
  
        #Store hyperparameter  
        self.learning_rate = learning_rate  
        self.momentum = momentum  
        self.reg_coeff = reg_coeff  
        self.dropout_rate = dropout_rate  
  
        #Network Architecture : 3 layer + BN!!  
        self.layer1 = nn.Linear(layer_dims[0], layer_dims[1])  
        self.bn1 = nn.BatchNorm1d(layer_dims[1])  
        self.layer2 = nn.Linear(layer_dims[1], layer_dims[2])  
        self.bn2 = nn.BatchNorm1d(layer_dims[2])  
        self.layer3 = nn.Linear(layer_dims[2], layer_dims[3])  
  
        #dropout layer for regularization  
        nn.dropout = nn.Dropout(p=self.dropout_rate)  
  
        #Parameter initialization  
        nn.init.kaiming_normal_(self.layer1.weight, nonlinearity='relu')  
        nn.init.kaiming_normal_(self.layer2.weight, nonlinearity='relu')  
        nn.init.kaiming_normal_(self.layer3.weight, nonlinearity='relu')  
  
        nn.init.zeros_(self.layer1.bias)  
        nn.init.zeros_(self.layer2.bias)  
        nn.init.zeros_(self.layer3.bias)
```

### 3-2. MODEL - Forward Pass Method
##### 1) input processing 
> accept input data and pass it through the network.
##### 2) [[activation function]]
- Apply appropriate activation functions (e.g., ReLU, Sigmoid, Softmax) at every hidden layer to introduce non-linearity
	- $\rightarrow$ [[ReLU series]] 사용
##### 3) intermediate computation 
- Save intermediate variables (if necessary) to be used later in the backward pass for computing gradients.
```python
def forward(self, x):  
	x = x.view(x.size(0), -1)  # Flatten the input image [batch_size, channels, height, width] -> [batch_size, input_dim]
    # Pass through the 1st linear layer, apply Batch Normalization, then ReLU and dropout  
    x = self.layer1(x)  
    x = self.bn1(x)  
    x = F.relu(x)  
    x = self.dropout(x)  
    # Pass through the 2nd linear layer, apply Batch Normalization, then ReLU and dropout  
    x = self.layer2(x)  
    x = self.bn2(x)  
    x = F.relu(x)  
    x = self.dropout(x)  
    # Pass through the 3rd linear layer, apply Batch Normalization, then ReLU and dropout  
    x = self.layer3(x)  
  
    return x
```
### 3-3. considerations 
##### 1) $\mathcal{L}(y, \hat{y})$ method
> calculate the discrepancy between the predicted outputs and the actual targets
> $\leftarrow$ [[loss function]]

$\rightarrow$ Binary [[Cross-Entropy Error]]
##### 2) for effectiveness
##### ✓early stop 
$\rightarrow$ 구현 : patience는 7로
##### ✓ [[regularization]] terms

##### ✓ dropout 
$\rightarrow$ [[neuron dropout regularization]] 

##### 3) for efficiency
###### ✓ [[optimization]] alg.
>Incorporate a method or integrate with an optimizer (e.g., SGD, Adam, RMSprop) that uses the computed gradients to update network parameters.

$\rightarrow$ [[Adam optimization alg.]] 사용
###### ✓ Learning Rate Scheduling
> Optionally include mechanisms to adjust the learning rate during training (e.g., step decay, exponential decay) to refine learning.

$\rightarrow$ epoch 수가 작아서 무의미

##### `train_one_epoch`
>Trains the model for one epoch.

- Args:
	  - model: The neural network model.
	  - dataloader: DataLoader providing the training data.
	  - criterion: Loss function $\rightarrow$ BCEWithLogitsLoss.
	  - optimizer: Adam optimizer with L2 regularization (weight_decay).
	  - device: 'cpu' or 'cuda'.
- Returns:
	  - avg_loss: Average training loss over the epoch.
	  - accuracy: Training accuracy over the epoch.

##### `evaluate_model`
>Evaluates the model on validation/test data.
- Args:
	  - model: The trained model.
	  - dataloader: DataLoader providing the evaluation data.
	  - criterion: Loss function.
	  - device: 'cpu' or 'cuda'.

- Returns:
	- avg_loss: Average evaluation loss.
	- accuracy: Proportion of correct predictions.

##### `train_model_with_early_stopping`
> Trains the model using early stopping based on validation loss.
  - Binary Cross-Entropy Error using BCEWithLogitsLoss.
  - Regularization (L2 via weight_decay in the optimizer) and dropout (in the model architecture).
  - Adam optimization algorithm.
  - Early stopping for effectiveness.
 - Learning rate scheduling is omitted (epoch count is small).

- Args
	- model: The PyTorch model.
	- train_loader: DataLoader for training data.
	- val_loader: DataLoader for validation data.
	- criterion: Loss function (BCEWithLogitsLoss recommended).
	- optimizer: Adam optimizer.
	- device: 'cpu' or 'cuda'.
	- num_epochs: Maximum number of epochs.
	- patience: Number of epochs with no improvement before stopping early.
- Returns:
	- model: The model loaded with the best validation parameters.



---
# 0x04. accuracy comparison and analysis
### 4-1. compare acc w/ bar plot  
![[epoch이 2일 때 LR vs DNN.png]]

![[epoch이 10일 때 LR vs DNN.png]]
![[epoch이 20일 때 LR vs DNN.png]]
### 4-2. the cause of accuracy difference 
##### LR
- LR 모델은 선형 모형임에 따라 데이터의 복잡한 패턴을 충분히 포착하지 못함으로 인해 epoch 수가 낮은 경우 학습이 부족(underfitting)함을 보임
	- → 낮은 epoch에서는 학습량이 부족하여 데이터 특성을 완벽히 반영하지 못함임
- LR 모델은 epoch 수가 증가할수록 training/validation 성능은 개선됨에도 불구하고, 모델의 단순성으로 인해 training 데이터에 과도하게 적합(overfitting)되어 test 데이터 성능이 오히려 저하됨
    - → 모델 파라미터 수가 제한적임에도 불구하고, 반복 학습 과정에서 노이즈나 데이터 특이사항까지 학습함으로써 일반화 성능이 감소함
##### DNN 
- DNN 모델은 다층 구조와 비선형 활성화를 통해 복잡한 패턴을 학습할 수 있음으로, 충분한 epoch과 Early stopping 적용했더니 training 성능과 test 성능 모두 개선됨
	- → 낮은 epoch에서는 학습량이 적어 underfitting 현상이 나타나지만, epoch 수가 적절히 증가하면 모델의 학습 역량이 발휘되어 좋은 일반화 성능을 얻음
- DNN 모델의 경우 Early stopping 기법이 적용되어 과적합을 효과적으로 방지함으로써, 높은 epoch 수에서도 test 성능이 오히려 향상됨
	- → 학습 과정 중 최적의 모델 시점을 포착하여 불필요한 반복 학습을 차단함
##### 왜 LR의 성능이 높을까?
- **데이터 전처리로 인한 단순화**
    - 엑스레이 사진은 그레이스케일로 변환되고 128×128 크기로 리사이즈되어 입력되므로, 이미지의 복잡도가 크게 낮아져 기본적인 픽셀 강도 정보만으로도 주요 특징이 어느 정도 보존됨. 이는 선형 모델인 LR도 중요한 정보를 효과적으로 학습할 수 있는 환경을 제공함.
- **특징의 선형 분리 가능성**
    - 폐렴으로 인해 엑스레이 상에서는 특정한 명암 변화나 구조적 이상이 발생하는 경우가 많아, 단순한 선형 결정 경계로도 충분히 분류가 가능하여 LR이 DNN과 유사한 분류 성능을 낸 것 같다.
- **모델 복잡도와 일반화**
    - LR은 파라미터 수가 적어 모델의 복잡도가 낮으면서도, 주어진 데이터셋이 선명한 특징을 내포하고 있는 경우 오버피팅 위험이 비교적 적고 일반화 능력이 우수함. 

### 4-3. train/val/test acc of epoch changes 
|**Model**|**Epoch**|**Train Accuracy**|**Validation Accuracy**|**Test Accuracy**|
|---|---|---|---|---|
|**LR**|**2**|~93.3%|90.3~93.5%|**81.25%**|
|**LR**|**10**|~94.5%|**95.7%**|**76.44%**|
|**LR**|**20**|~95.6%|**95.4%**|**76.12%**|
|**DNN**|**2**|**94.99%**|**94.25%**|**69.55%**|
|**DNN**|**10**|**97.27%**|**94.92%**|**79.33%**|
|**DNN**|**20**|**97.77%**|**96.74%**|**84.13%**|
### 4-4. Analysis of underfitting/overfitting due to epoch changes
| **Model** | **Epoch** | **상태**              | **비고**                           |
| --------- | --------- | ------------------- | -------------------------------- |
| LR        | 2         | **Underfitting**    | 학습 부족으로 성능이 불안정함                 |
| LR        | 10        | **Overfitting**     | Validation 성능에 비해 Test 성능 크게 저하됨 |
| LR        | 20        | **심화된 Overfitting** | Test 성능 추가 하락, overfitting 심화됨   |
| DNN       | 2         | **Underfitting**    | Epoch 부족으로 Test 성능 매우 낮음         |
| DNN       | 10        | **경미한 Overfitting** | Early stopping으로 다소 완화           |
| DNN       | 20        | **적절한 학습 상태**       | Early stopping이 과적합 방지하여 최적의 성능  |
