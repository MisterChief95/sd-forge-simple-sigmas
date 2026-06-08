"""
Simple Sigmas — override sigma schedule parameters per generation.

Sigma min/max are applied directly to the model's predictor on a cloned UNet,
so changes are scoped to the current generation and never affect the loaded model.
Rho and beta distribution parameters are patched into shared opts for the duration
of sampling and restored in postprocess.

A value of 0 for any numeric field means "use the model/scheduler default" and
will be skipped entirely.
"""

import gradio as gr

from modules import scripts
from modules.infotext_utils import PasteField
from modules.ui_components import InputAccordion

from modules.script_callbacks import remove_callbacks_for_function, on_script_unloaded
from modules.shared import opts


class SimpleSigmas(scripts.Script):
    sorting_priority = 15

    def __init__(self):
        super().__init__()
        self._orig: dict[str, float] = {}

    def _restore_opts(self):
        """Restore patched opts to their pre-generation values.

        Called both in postprocess and on extension hot-reload via on_script_unloaded,
        so a crashed or interrupted generation doesn't leave opts in a dirty state.
        """
        if not self._orig:
            return
        opts.data.update(self._orig)
        self._orig = {}

    def _save_orig(self):
        """Snapshot the current opts values we intend to patch (idempotent)."""
        defaults = {"rho": 0.0, "beta_dist_alpha": 0.6, "beta_dist_beta": 0.6}
        for key, default in defaults.items():
            if key not in self._orig:
                self._orig[key] = opts.data.get(key, default) or default

    def _patch_opts(self, rho, beta_dist_alpha, beta_dist_beta):
        """Write override values into shared opts for the duration of sampling."""
        updates = {}
        if rho != 0:
            updates["rho"] = rho
        if beta_dist_alpha != 0:
            updates["beta_dist_alpha"] = beta_dist_alpha
        if beta_dist_beta != 0:
            updates["beta_dist_beta"] = beta_dist_beta
        if updates:
            opts.data.update(**updates)

    @staticmethod
    def _apply_sigma_range(unet, sigma_min, sigma_max):
        """Apply sigma_min / sigma_max overrides to the predictor on a cloned UNet.

        EDM-style predictors (PredictionContinuousEDM, PredictionContinuousV) use a
        direct formula for timestep(), so their sigmas buffer can be freely replaced
        via set_parameters().

        Discrete/beta-schedule predictors (Prediction, PredictionEDM) use a
        nearest-neighbour lookup into log_sigmas to map sigma → conditioning timestep.
        Replacing the full buffer remaps those lookups and corrupts the UNet's
        conditioning, causing instability. For these we only move the endpoint entries,
        leaving the interior lookup table intact.

        A value of 0 for either bound means "keep the model default" and is skipped.
        """
        import torch
        predictor = unet.model.predictor

        if hasattr(predictor, 'set_parameters'):
            # EDM continuous: safe to rebuild the whole buffer
            new_min = sigma_min if sigma_min != 0 else predictor.sigma_min.item()
            new_max = sigma_max if sigma_max != 0 else predictor.sigma_max.item()
            predictor.set_parameters(new_min, new_max, predictor.sigma_data)
        else:
            # Discrete: only shift the endpoints, preserve interior
            s = predictor.sigmas
            if sigma_min != 0:
                s[0] = torch.tensor(sigma_min, dtype=s.dtype, device=s.device).clamp(max=s[0])
                predictor.log_sigmas[0] = s[0].log()
            if sigma_max != 0:
                s[-1] = torch.tensor(sigma_max, dtype=s.dtype, device=s.device).clamp(min=s[-1])
                predictor.log_sigmas[-1] = s[-1].log()

    def title(self):
        return "Simple Sigmas"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(False, label="Simple Sigmas") as enabled:
            gr.Markdown(
                "Override the sigma schedule used during sampling. "
                "**Default value of `0` is ignored.**\n\n"
                "- **Sigma min/max** — clamp the noise schedule range on the model predictor directly, "
                "compatible with all schedulers.\n"
                "- **Rho** — controls the shape of the Karras and Polyexponential schedules "
                "(higher = more steps near sigma_min).\n"
                "- **Beta schedule** — shapes the Beta scheduler's timestep distribution; "
                "alpha/beta < 1 concentrates steps at the ends, > 1 concentrates in the middle."
            )
            with gr.Group():
                with gr.Tab("Base"):
                    with gr.Group():
                        gr.Markdown("Sigma range")
                        with gr.Row():
                            sigma_min = gr.Number(
                                label="Sigma min (0 = model default)",
                                value=0.0,
                                minimum=0.0,
                                precision=4,
                            )
                            sigma_max = gr.Number(
                                label="Sigma max (0 = model default)",
                                value=0.0,
                                minimum=0.0,
                                precision=4,
                            )

                    with gr.Group():
                        gr.Markdown("Karras / Polyexponential")
                        with gr.Row():
                            rho = gr.Number(
                                label="Rho (0 = scheduler default)",
                                value=0.0,
                                minimum=0.0,
                                precision=4,
                            )

                    with gr.Group():
                        gr.Markdown("Beta schedule")
                        with gr.Row():
                            beta_alpha = gr.Slider(
                                label="Beta alpha (0 = scheduler default)",
                                value=0.0,
                                minimum=0.0,
                                maximum=2.0,
                                step=0.01,
                            )
                            beta_beta = gr.Slider(
                                label="Beta beta (0 = scheduler default)",
                                value=0.0,
                                minimum=0.0,
                                maximum=2.0,
                                step=0.01,
                            )

                with gr.Tab("Hires Fix"):
                    with gr.Group():
                        hr_override = gr.Checkbox(label="Use separate values for Hires Fix", value=False)

                    with gr.Group():
                        gr.Markdown("Sigma range")
                        with gr.Row():
                            hr_sigma_min = gr.Number(
                                label="Sigma min (0 = base value)",
                                value=0.0,
                                minimum=0.0,
                                precision=4,
                            )
                            hr_sigma_max = gr.Number(
                                label="Sigma max (0 = base value)",
                                value=0.0,
                                minimum=0.0,
                                precision=4,
                            )

                    with gr.Group():
                        gr.Markdown("Karras / Polyexponential")
                        with gr.Row():
                            hr_rho = gr.Number(
                                label="Rho (0 = base value)",
                                value=0.0,
                                minimum=0.0,
                                precision=4,
                            )

        self.infotext_fields = [
            PasteField(enabled, "SS Enabled", api="ss_enabled"),
            PasteField(sigma_min, "SS Sigma Min", api="ss_sigma_min"),
            PasteField(sigma_max, "SS Sigma Max", api="ss_sigma_max"),
            PasteField(rho, "SS Rho", api="ss_rho"),
            PasteField(beta_alpha, "SS Beta Alpha", api="ss_beta_alpha"),
            PasteField(beta_beta, "SS Beta Beta", api="ss_beta_beta"),
            PasteField(hr_override, "SS HR Override", api="ss_hr_override"),
            PasteField(hr_sigma_min, "SS HR Sigma Min", api="ss_hr_sigma_min"),
            PasteField(hr_sigma_max, "SS HR Sigma Max", api="ss_hr_sigma_max"),
            PasteField(hr_rho, "SS HR Rho", api="ss_hr_rho"),
        ]

        return [enabled, sigma_min, sigma_max, rho, beta_alpha, beta_beta,
                hr_override, hr_sigma_min, hr_sigma_max, hr_rho]

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        enabled, sigma_min, sigma_max, rho, beta_alpha, beta_beta, \
            hr_override, hr_sigma_min, hr_sigma_max, hr_rho = script_args

        if not enabled:
            return

        is_hr = getattr(p, "is_hr_pass", False)

        # Pick the active values for this pass; HR values of 0 fall back to main values
        if is_hr and hr_override:
            _sigma_min = hr_sigma_min if hr_sigma_min != 0 else sigma_min
            _sigma_max = hr_sigma_max if hr_sigma_max != 0 else sigma_max
            _rho       = hr_rho       if hr_rho       != 0 else rho
        else:
            _sigma_min, _sigma_max, _rho = sigma_min, sigma_max, rho

        # Apply sigma range on a cloned UNet — no postprocess revert needed
        if _sigma_min != 0 or _sigma_max != 0:
            unet = p.sd_model.forge_objects.unet.clone()
            self._apply_sigma_range(unet, _sigma_min, _sigma_max)
            p.sd_model.forge_objects.unet = unet

        # Patch opts for rho and beta (restored in postprocess)
        on_script_unloaded(self._restore_opts)
        self._save_orig()
        self._patch_opts(rho=_rho, beta_dist_alpha=beta_alpha, beta_dist_beta=beta_beta)

        # Write infotext — only non-zero values, using standard keys where applicable
        p.extra_generation_params["SS Enabled"] = True
        if is_hr:
            if _sigma_min != 0:
                p.extra_generation_params["SS HR Sigma Min"] = _sigma_min
            if _sigma_max != 0:
                p.extra_generation_params["SS HR Sigma Max"] = _sigma_max
            if _rho != 0 or p.scheduler not in ["Karras", "Polyexponential"]:
                p.extra_generation_params["SS HR Rho"] = _rho
        else:
            if _sigma_min != 0:
                p.extra_generation_params["Schedule min sigma"] = _sigma_min
                p.extra_generation_params["SS Sigma Min"] = _sigma_min
            if _sigma_max != 0:
                p.extra_generation_params["Schedule max sigma"] = _sigma_max
                p.extra_generation_params["SS Sigma Max"] = _sigma_max
            if _rho != 0 or p.scheduler not in ["Karras", "Polyexponential"]:
                p.extra_generation_params["Schedule rho"] = _rho
                p.extra_generation_params["SS Rho"] = _rho
        if beta_alpha != 0:
            p.extra_generation_params["SS Beta Alpha"] = beta_alpha
        if beta_beta != 0:
            p.extra_generation_params["SS Beta Beta"] = beta_beta

    def postprocess(self, p, processed, enabled, sigma_min, sigma_max, rho, beta_alpha, beta_beta,
                    hr_override, hr_sigma_min, hr_sigma_max, hr_rho, *args):
        remove_callbacks_for_function(self._restore_opts)
        if not enabled:
            return
        self._restore_opts()
