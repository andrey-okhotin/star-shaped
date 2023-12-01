from diffusion.image_data_diffusion.ddpm_diffusion import DDPM

    


class SyntheticDDPM(DDPM):

    def model_prediction(self, model, xt, t):
        normed_xt = self.time_dependent_xt_normalization(xt, t)
        rescaled_t = self.time_rescaler(t)
        x0_pred = model(normed_xt, rescaled_t)
        return x0_pred

