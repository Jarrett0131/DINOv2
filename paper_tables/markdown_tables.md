## 不同模型/方法 Accuracy 对比

| Dataset | Backbone      | Method              | Top-1 Accuracy (%) | Feature Dim |
| ------- | ------------- | ------------------- | ------------------ | ----------- |
| CUB     | dinov2_vitb14 | Linear Probe        | 88.45              | 768         |
| CUB     | dinov2_vitb14 | k-NN (cosine, k=10) | 85.57              | 768         |

## Linear Probe 与 k-NN 对比

| Dataset | Linear Probe (%) | k-NN (%) | Delta (%) |
| ------- | ---------------- | -------- | --------- |
| CUB     | 88.45            | 85.57    | 2.88      |

## 参数量对比

| Model                    | Parameters (M) | Trainable in Linear Probe (M) | Source                              |
| ------------------------ | -------------- | ----------------------------- | ----------------------------------- |
| DINOv2 ViT-B/14 backbone | 86.58          | 0.00                          | Measured from torch hub model       |
| Linear classifier head   | 0.15           | 0.15                          | 768 x 200 + 200 checkpoint metadata |

## 推理速度对比

| Backbone      | Device | Batch Size | Images/s | Latency (ms/image) | Note                                                                                                   |
| ------------- | ------ | ---------- | -------- | ------------------ | ------------------------------------------------------------------------------------------------------ |
| dinov2_vitb14 | cuda   | 64         | 32.56    | 30.71              | Measured on 20 test batches after 2 warmup batches; includes image transfer and backbone forward only. |

## 不同数据集结果

| Dataset | Backbone      | Method              | Top-1 Accuracy (%) |
| ------- | ------------- | ------------------- | ------------------ |
| CUB     | dinov2_vitb14 | Linear Probe        | 88.45              |
| CUB     | dinov2_vitb14 | k-NN (cosine, k=10) | 85.57              |
