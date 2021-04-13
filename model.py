import torch
import numpy as np
import torch.nn as nn

def noise_schedule(T,s=0.008):
    """
    a improved beta generator
    from https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = T + 1
    t = torch.linspace(0, steps, steps)
    f = torch.cos((t/steps+s)*np.pi/(2+2*s))**2
    alphas_bar = f / f[0]
    betas = 1 - (alphas_bar[1:]/alphas_bar[:-1])
    return torch.clip(betas, min = 0, max = 0.999)

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

class DenoisingDiffusionNet(nn.Module):
    def __init__(self,num_timesteps,denoise_model,eta = 0):
        super(DenoisingDiffusionNet, self).__init__()
        self.denoise_model = denoise_model
        self.num_timesteps = num_timesteps# denoising和diffusion的步数
        self.eta = eta
        
        betas = noise_schedule(self.num_timesteps)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_previous = torch.cat([torch.tensor([1.0]),alphas_bar[:-1]])
        sqrt_alphas_bar = torch.sqrt(alphas_bar)
        sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - alphas_bar)
        sqrt_recip_minus_one_alphas_bar = torch.sqrt(1.0 / alphas_bar - 1.0)
        
        coef1 = self.eta * torch.sqrt((1-alphas_bar_previous)/(1-alphas_bar) * (1-alphas_bar/alphas_bar_previous))
        coef2 = torch.sqrt(1-alphas_bar_previous-coef1**2)
        
        
        self.register_buffer('betas',betas)
        self.register_buffer('alphas',alphas)
        self.register_buffer('alphas_bar',alphas_bar) #actually alpha_t_bar
        self.register_buffer('sqrt_alphas_bar',sqrt_alphas_bar)
        self.register_buffer('sqrt_one_minus_alphas_bar',sqrt_one_minus_alphas_bar)#sqrt(1-alphas_t_bar)
        self.register_buffer('alphas_bar_previous',alphas_bar_previous)# alpha_t-1_bar
        self.register_buffer('sqrt_recip_minus_one_alphas_bar',sqrt_recip_minus_one_alphas_bar)#sqrt(1/alphas_t_bar - 1)
        self.register_buffer('coef1',coef1)
        self.register_buffer('coef2',coef2)
        
        
        
    def q(self,img,t,noise):
        return (extract(self.sqrt_alphas_bar, t, img.shape) * img + \
            extract(self.sqrt_one_minus_alphas_bar, t, img.shape) * noise)
        
        
    def sample(self,device,eta = 0):
        with torch.no_grad():
            x = torch.randn(1,3,32,32,device=device)
            seq = range(0, self.num_timesteps, 1)

            n = 1
            seq_next = [-1] + list(seq[:-1])
            x0_preds = []
            xs = [x]
            for i, j in zip(reversed(seq), reversed(seq_next)):
                t = (torch.ones(n) * i).long().to(device)
                next_t = (torch.ones(n) * j).long().to(device)

                xt = xs[-1].to(device)
                et = self.denoise_model(xt, t)

                x0_t =  extract(1.0 / self.sqrt_alphas_bar, t, xt.shape) * xt - \
                    extract(self.sqrt_recip_minus_one_alphas_bar, t, xt.shape) * et         
                x0_preds.append(x0_t.to('cpu'))


                xt_next = extract(self.alphas_bar_previous.sqrt(), t, xt.shape) * x0_t + \
                        extract(self.coef2, t, xt.shape) * et + \
                        extract(self.coef1, t, xt.shape) * torch.randn_like(x)

                xs.append(xt_next.to('cpu'))

            return xs[-1][0]
            
            
                
    def forward(self,x):
        batch_size = x.shape[0]
        t = torch.randint(0,self.num_timesteps,(batch_size,),device=x.device).long() # randomly generate t(for each batch)
        noise = torch.randn_like(x)
        img_noise = self.q(x,t,noise) # construct noise image
        img_reconstruct = self.denoise_model(img_noise,t)
        loss = (img_noise - img_reconstruct).abs().mean()
        return loss
        
