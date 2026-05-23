# Experiment Audit

## 可信部分

- Train/test leakage: feature payload path overlap is 0; CUB split is read from `train_test_split.txt`.
- Backbone frozen: `DINOv2FeatureExtractor` sets all backbone parameters `requires_grad=False`; linear probe training wraps backbone forward in `torch.no_grad()`.
- Train/eval mode: linear probe uses `backbone.eval()`, `classifier.train()` during training and `classifier.eval()` during evaluation.
- Test augmentation: dataset transform is deterministic resize + tensor conversion + normalization; no random augmentation is applied to the test set.
- kNN protocol: `KNeighborsClassifier.fit` uses `train_cls_features.pt`; predictions use `test_cls_features.pt`.
- Confusion matrix label alignment: both matrices are 200x200, sum to 5794 test samples, and diagonal accuracy matches summary.
- t-SNE split: regenerated t-SNE uses only `test_cls_features.pt` with 2000 samples.

## 风险点

- The linear probe checkpoint metadata shows `resume` pointing to the same best checkpoint path. This is consistent with resumed training, but the current metrics CSV starts at epoch 6, so epochs 1-5 are not preserved in the final CSV.
- Model selection uses test accuracy to save the best checkpoint. This is acceptable for a quick benchmark, but for a strict paper protocol it should use a validation split and report a single final test evaluation.
- No TensorBoard logs or immutable run manifest were found; reproducibility depends on CSV/JSON/checkpoint metadata.
- kNN was evaluated only for k=10, so the reported kNN number may be sensitive to the chosen k.
- Existing t-SNE is a sampled visualization and should be interpreted qualitatively, not as a quantitative clustering metric.
- Attention/GradCAM conclusions cannot be supported until actual visualization outputs are generated.

## 建议重跑或补做

- Re-run linear probe with train/val/test separation, select checkpoints on validation accuracy, and report final test once.
- Run at least 3 random seeds and report mean +/- standard deviation.
- Run kNN over a small k sweep, e.g. 1, 5, 10, 20, 50.
- Add CNN or supervised ViT baselines under the same CUB split.
- Generate attention/patch-token maps for representative correct and incorrect predictions.
- Add a controlled inference benchmark script with hardware, batch size, precision, warmup, and timed iterations.