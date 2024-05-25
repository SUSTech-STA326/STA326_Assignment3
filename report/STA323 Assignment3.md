# STA323 Assignment3

## 12111641 魏悦阳

### Requirements

![image-20240525105528257](assets/image-20240525105528257.png)

### Model description

| Model                                                        | Summary                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="assets/v2-748f6c066e2d8782b38fb977347f0d04_r.jpg" alt="img" style="zoom:67%;" /> | ![image-20240525110658868](assets/image-20240525110658868.png) |
| <img src="assets/v2-82c2474d1e79d50ff9179c83a6821779_720w.webp" alt="img" style="zoom:67%;" /> | ![image-20240525133639537](assets/image-20240525133639537.png) |
| <img src="assets/v2-c23343c0a3800dabfe0f523a170b10a2_720w.webp" alt="img" style="zoom:67%;" /> | ![image-20240525110731290](assets/image-20240525110731290.png) |

### Parameters

| Parameter            | Set                  |
| -------------------- | -------------------- |
| **num of negatives** | 4                    |
| **learner**          | Adam                 |
| **learning rate**    | 0.001                |
| **loss function**    | binary cross entropy |
| **batch size**       | 2048                 |
| **mlp layer**        | 3                    |
| **epoch**            | 50                   |
| **factor number**    | 8                    |

### Evaluate

| Name                                        | Function                                                     |
| ------------------------------------------- | ------------------------------------------------------------ |
| HR(Hits Ratio)                              | $\mathrm{HR}=\frac1{\mathrm{N}}\sum_{\mathrm{i}=1}^{\mathrm{N}}\mathrm{hits}(\mathrm{i})$ |
| NDCG(Normalized Discounted Cumulative Gain) | $\mathrm{NDCG}=\frac{1}{\mathrm{N}}\sum_{{\mathrm{i}=1}}^{{\mathrm{N}}}\frac{1}{\log_{2}\left(\mathrm{p_{i}}+1\right)}$ |

### Experiments

I will point out different parameters comparing with the list above.

#### Model Compare

| Training Loss                                                | HR@10                                                        | NDCG@10                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20240525154400488](assets/image-20240525154400488.png) | ![image-20240525154406046](assets/image-20240525154406046.png) | ![image-20240525154410792](assets/image-20240525154410792.png) |

#### Ablation study for the MLP layer

For this part, I use **epoch** of 5, **mlp layer** from 0 to 4.

| Training Loss                                                | HR@10                                                        | NDCG@10                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20240525170225357](assets/image-20240525170225357.png) | ![image-20240525170230185](assets/image-20240525170230185.png) | ![image-20240525170238479](assets/image-20240525170238479.png) |

| Evaluate    | MLP-0 | MLP-1 | MLP-2 | MLP-3 | MLP-4     |
| ----------- | ----- | ----- | ----- | ----- | --------- |
| **HR@10**   | 0.544 | 0.561 | 0.595 | 0.614 | **0.642** |
| **NDCG@10** | 0.385 | 0.394 | 0.404 | 0.418 | **0.427** |

#### Ablation study for different topK

For this part, I use **epoch** of 5, **topK** from 1 to 10.

| Training Loss                                                | HR@10                                                        | NDCG@10                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20240525174009803](assets/image-20240525174009803.png) | ![image-20240525174150027](assets/image-20240525174150027.png) | ![image-20240525174212463](assets/image-20240525174212463.png) |

| HR                                                           | NDCG                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20240525174354443](assets/image-20240525174354443.png) | ![image-20240525174349950](assets/image-20240525174349950.png) |

#### Ablation study for different factor

For this part, I use **epoch** of 5, **factor number** of 8, 16, 32, 64.

| Training Loss                                                | HR@10                                                        | NDCG@10                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20240525173813336](assets/image-20240525173813336.png) | ![image-20240525173817390](assets/image-20240525173817390.png) | ![image-20240525173821078](assets/image-20240525173821078.png) |

| HR@10                                                        | NDCG@10                                                      |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![image-20240525173727688](assets/image-20240525173727688.png) | ![image-20240525173731880](assets/image-20240525173731880.png) |

