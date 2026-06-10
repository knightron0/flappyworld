# Final

This directory is organized around the current browser model pipeline.

## Folders

- `scripts/`
  - Runnable entrypoints.
- `model/`
  - Shared model, export, preprocessing, and player support code.
- `web/`
  - Browser app, assets, and exported model files.
- `cluster/`
  - Optional batch scripts for collect/preprocess/train.
- `flappy_bird_gymnasium/`
  - Environment package used for PPO data collection.

## Main commands

Run these from `final/`.

1. Train PPO if needed:
   - `python -m scripts.train_policy`
2. Collect raw episodes:
   - `python -m scripts.collect_data ...`
3. Preprocess dataset:
   - `python -m scripts.preprocess_data ...`
4. Train model:
   - `python -m scripts.train_model ...`
5. Export ONNX:
   - `python -m scripts.export_model export ...`
6. Export browser manifest:
   - `python -m scripts.export_manifest ...`
7. Local player:
   - `python -m scripts.play ...`
8. Episode renderer:
   - `python -m scripts.render_episode ...`

## Notes

- `scripts/` is the intended surface. `model/` is support code.
- `web/` includes the current `public/model/` and `public/flappy-assets/`.
- `cluster/` mirrors the same collect/preprocess/train flow with renamed job files.
