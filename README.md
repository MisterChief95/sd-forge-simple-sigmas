
# Simple Sigmas

Override sigma schedule parameters per generation in Stable Diffusion WebUI Forge.

## Features

- **Sigma min/max** — Clamp the noise schedule range directly on the model predictor. Compatible with all schedulers.
- **Rho** — Controls the shape of Karras and Polyexponential schedules. Higher values concentrate more steps near sigma_min.
- **Beta schedule** — Shape the Beta scheduler's timestep distribution. Values < 1 concentrate steps at the ends; > 1 concentrates in the middle.
- **Hires Fix overrides** — Use separate sigma parameters during upscaling passes.

## Usage

1. Enable the **Simple Sigmas** accordion in the UI
2. Enter desired values (0 = use model default and will be skipped)
3. Optionally enable **Hires Fix overrides** for separate upscale parameters
4. Generate normally

## Technical Details

- Sigma min/max overrides are applied to a cloned UNet, so changes never affect the loaded model
- Rho and beta distribution parameters are patched into shared options during sampling and restored afterward
- EDM-style predictors use full buffer replacement; discrete predictors only move endpoint entries to preserve interior lookup tables
