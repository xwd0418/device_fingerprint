from models.PL_resnet import *
from models.model_factory import *
import torchvision
from models.dc1d.dc1d.nn import DeformConv1d
import scipy, time
import matplotlib.pyplot as plt


class RandConv(Baseline_Resnet):
    def __init__(self, config):
        super().__init__(config)
        self.tanh = nn.Tanh()
        self.num_channels, self.data_dimension = (3,2) if self.config['dataset'].get("img") else (2,1)
        data_sample_size = 224 if self.config['dataset'].get("img") else 256
        self.kernel_size = 3
        self.instance_norm = torch.nn.InstanceNorm2d(self.num_channels) if self.config['dataset'].get("img")\
            else torch.nn.InstanceNorm2d(self.num_channels)
        #Gaussian radom field
        if self.config['model']['conv_type'] == "block":        
            GRF_coeff = config['model']['GRF_coeff']
            self.register_buffer("amplitude" , torch.from_numpy(get_noise_amplitude(data_sample_size, self.data_dimension, GRF_coeff)))
            self.GRF_nums = self.data_dimension * (self.kernel_size**self.data_dimension)
            
            batch_size = self.config['dataset']['batch_size']
            if self.config['dataset'].get("img"):
                batch_size = batch_size // 3 * 3
            random_shape =  [batch_size, self.GRF_nums]+[data_sample_size for _ in range(self.data_dimension)]
            self.register_buffer("noise_place_holder_1", torch.empty(size = random_shape))
            self.register_buffer("noise_place_holder_2", torch.empty(size = random_shape))
            self.GRF_norm = torch.nn.InstanceNorm2d(self.GRF_nums)
            
        self.init_rand_conv_layer()
        
    def training_step(self, batch, batch_idx):
        x, y = self.unpack_batch(batch)
        # print(x.shape, "\n\n\n\n\n")
        num_reptition = np.random.randint(1, 1+self.config['model']["max_conv_reptitions"]) #  +1 because of exclusiveness on the 2nd parameter 
        rand_conved_x = self.rand_conv(x,num_reptition)
        # print("rand_conved_x", rand_conved_x.shape, "\n\n\n\n\n")
        
        logits = self(rand_conved_x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, y)
        self.log("train/loss", loss,  prog_bar=False, on_step=not self.train_log_on_epoch,
                 on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1)
        self.log("train/acc", self.train_accuracy, prog_bar=True, on_step=not self.train_log_on_epoch, 
                 on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1)   

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = self.unpack_batch(batch, single_domain_loader=True)
        x = self.rand_conv(x,num_reptition=1)
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy(preds, y)
        self.log("val/loss", loss, prog_bar=False, sync_dist=torch.cuda.device_count()>1)
        self.log("val/acc", self.val_accuracy, prog_bar=True, sync_dist=torch.cuda.device_count()>1)
        
    def test_step(self, batch, batch_idx):
        x, y = self.unpack_batch(batch, single_domain_loader=True)
        x = self.rand_conv(x,num_reptition=1)
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test/loss", loss, prog_bar=False, sync_dist=torch.cuda.device_count()>1)
        self.log("test/acc", self.test_accuracy, prog_bar=True, sync_dist=torch.cuda.device_count()>1)
        # return self.test_accuracy.compute()
        # return 0
    
    @torch.no_grad()
    def init_rand_conv_layer(self):
        # using default kaiming's initiliation, according to the rand_conv paper
        if self.config['model']['conv_type'] == "plain":
            conv = nn.Conv2d if self.config['dataset'].get("img") else nn.Conv1d
            self.rand_conv_layer = conv(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=self.kernel_size, padding="same")
        elif self.config['model']['conv_type'] == "block":
            conv = torchvision.ops.DeformConv2d if self.config['dataset'].get("img") else DeformConv1d
            # conv = nn.Conv1d
            self.rand_conv_layer = conv(in_channels=self.num_channels, out_channels=self.num_channels, kernel_size=self.kernel_size, padding="same")
        else :
            raise Exception("what kind of random  to use?")
    
    @torch.no_grad()
    def rand_conv(self, x, num_reptition):        
        # initializa distribution parameters
        upper_bound_gaussian_smoothing_sigma = self.config['model']['upper_bound_gaussian_smoothing_sigma']
        sigma_gauss_filter = np.random.uniform(1e-7, upper_bound_gaussian_smoothing_sigma)
        # initialize random weights 
        gaussian_filter = fspecial_gauss(self.kernel_size, sigma_gauss_filter, dimension=self.data_dimension)
        gaussian_filter = torch.from_numpy(gaussian_filter).to(x)
        self.rand_conv_layer.weight =  torch.nn.Parameter(self.rand_conv_layer.weight * gaussian_filter)
        
        
        if self.config['model']['conv_type'] == "plain":
            for _ in range(num_reptition):
                x = self.rand_conv_layer(x)
        elif self.config['model']['conv_type'] == "block":   
            """   parameter initialize   """  
            # 1. deformable conv offset  
            upper_bound_distortion_sigma = self.config['model']['upper_bound_distortion_sigma']
            distortion_scale = np.random.uniform(1e-7, upper_bound_distortion_sigma)
            
            offset = self.gaussian_random_field(scale=distortion_scale).to(x)
            if offset.shape[0] != x.shape[0]: # last batch
                offset = offset[0:x.shape[0]]
            # offset = np.random.normal(size = (63, GRF_nums , 224, 224))
            offset = self.GRF_norm(offset)
            
            if self.config['dataset'].get("img"):
                offset_name = "offset"  
            else :
                offset = offset.transpose(1,2).unsqueeze(1)
                offset_name = "offsets"
                
            kwargs = {offset_name:offset} if self.config['model']['conv_type'] == "block" else {}
        
            # 2. affine tranform
            # sigma_affine_scale = self.config['model']['sigma_affine_scale']
            sigma_affine_offset = self.config['model']['sigma_affine_offset']
            # affine_mean = np.random.normal(0,sigma_affine_scale)
            affine_offset = np.random.normal(0,sigma_affine_offset)
            # for number in range(3):
            #     save_img(x, number, f"inspect_orig_{number}.png", True)
            for _ in range(num_reptition):
                x = self.rand_conv_layer(x, **kwargs) # deformable conv
                x = self.instance_norm(x)# standardize
                x = x + affine_offset # affine tranform 
                # note that we do not use affine_mean here 
                x = self.tanh(x) # tanh
            # for number in range(3):
                # save_img(x, number, f"inspect_conved_{number}.png")
        return x
    


    def gaussian_random_field(self, scale = 1):
        # https://github.com/bsciolla/gaussian-random-fields/blob/master/gaussian_random_fields.py
        """ Returns a np array of shifted Fourier coordinates k_x k_y.
            
            Input args:
                alpha (double, default = 3.0): 
                    The power of the power-law momentum distribution
                size (integer, default = 128):
                    The size of the square output Gaussian Random Fields
                flag_normalize (boolean, default = True):
                    Normalizes the Gaussian Field:
                        - to have an average of 0.0
                        - to have a standard deviation of 1.0

            Returns:
                gfield (np array of shape (size, size)):
                    The random gaussian random field
                    
            Example:
            import matplotlib
            import matplotlib.pyplot as plt
            example = gaussian_random_field()
            plt.imshow(example)
            """
            
        
            # Draws a complex gaussian random noise with normal
            # (circular) distribution
        
        
        noise = self.noise_place_holder_1.normal_(std=scale) \
            + 1j * self.noise_place_holder_2.normal_(std=scale)
            # To real space
        gfield = torch.fft.ifft2(noise * self.amplitude).real
        
            # Sets the standard deviation to one        
        return gfield


# helper functions 
def get_noise_amplitude(size, dimension, alpha):
    """ compute a np array of shifted Fourier coordinates k_x k_y.
        and return an amplitude map for the random noise in GRF
        Input args:
            size (integer): The size of the coordinate array to create
        Returns:
            k_ind, np array of shape (2, size, size) with:
                k_ind[0,:,:]:  k_x components
                k_ind[1,:,:]:  k_y components
                
        Example:
        
            print(fftind(5))
            
            [[[ 0  1 -3 -2 -1]
            [ 0  1 -3 -2 -1]
            [ 0  1 -3 -2 -1]
            [ 0  1 -3 -2 -1]
            [ 0  1 -3 -2 -1]]

            [[ 0  0  0  0  0]
            [ 1  1  1  1  1]
            [-3 -3 -3 -3 -3]
            [-2 -2 -2 -2 -2]
            [-1 -1 -1 -1 -1]]]
            
        """
    if dimension == 2:
        # Defines momentum indices
        k_ind = np.mgrid[:size, :size] - int( (size + 1)/2 )
        k_ind = scipy.fftpack.fftshift(k_ind)
            # Defines the amplitude as a power law 1/|k|^(alpha/2)
        amplitude = np.power( k_ind[0]**2 + k_ind[1]**2 + 1e-10, -alpha/4.0 )
        amplitude[0,0] = 0
        
    if dimension == 1:
        # Defines momentum indices
        k_ind = np.mgrid[:size] - int( (size + 1)/2 )
        k_ind = scipy.fftpack.fftshift(k_ind)       
            # Defines the amplitude as a power law 1/|k|^(alpha/2)
        amplitude = np.power( k_ind**2 + 1e-10, -alpha/4.0 )
        amplitude[0] = 0
    return amplitude 
    
def fspecial_gauss(size, sigma, dimension):
    #  https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    if dimension == 2:
        x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    elif dimension == 1:
        g = scipy.signal.windows.gaussian(size, sigma)
    return g


def save_img(x, i, path, need_long=False):
    # im = x[i].permute(1,2,0).detach().cpu()
    # if need_long:
    #     im = im.long()
    # plt.figure()
    # plt.imshow(im[:,:,[2,1,0]])
    # plt.savefig(path)
    
    im = x[i,0].detach().cpu()
    plt.figure()
    plt.plot(im)
    plt.savefig(path)