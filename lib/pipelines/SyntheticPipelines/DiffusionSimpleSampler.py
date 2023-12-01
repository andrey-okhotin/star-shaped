import torch




def simple_sampler(
    model,
    diffusion,
    num_samples,
    batch_size,
    dequantizator=None,
    device=None
):
    prev_device = model.device
    if device is None:
        device = model.device
    model.to(device)
    diffusion.to(device)
    
    batch = {
        'batch_size' : batch_size,
        'device' : device
    }
    with torch.no_grad():
        model.eval()
        samples = []
        while len(samples) * batch_size < num_samples:
            try:
                result_objects = diffusion.sampling_procedure(
                    batch=batch,
                    model=model,
                    progress_printer=lambda t: t, 
                    num_sampling_steps=-1
                )
                samples.append(diffusion.from_domain(result_objects))
            except ValueError:
                print(" <<< VALUE ERROR >>> ")
                pass
        
    model.to(prev_device)
    diffusion.to(prev_device)
    samples = torch.vstack(samples)[:num_samples]
    return samples



