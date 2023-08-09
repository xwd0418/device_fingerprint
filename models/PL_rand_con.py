from models.PL_resnet import *
from models.model_factory import *
import torchvision
from models.dc1d.dc1d.nn import DeformConv1d
import scipy
from torchvision import transforms

class RandConv(Baseline_Resnet):
    def __init__(self, config):
        super().__init__(config)
        self.tanh = nn.Tanh()
        num_channels = 3 if self.config['dataset'].get("img") else 2
        self.instance_norm = torch.nn.InstanceNorm2d(num_channels)
    
    def training_step(self, batch, batch_idx):
        x, y = self.unpack_batch(batch)
        
        rand_conved_x = self.rand_conv(x)
        logits = self(rand_conved_x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, y)
        self.log("train/loss", loss,  prog_bar=False, on_step=not self.train_log_on_epoch,
                 on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1)
        self.log("train/acc", self.train_accuracy, prog_bar=True, on_step=not self.train_log_on_epoch, 
                 on_epoch=self.train_log_on_epoch, sync_dist=torch.cuda.device_count()>1)   

        return loss
    
    def rand_conv(self, x):
        num_channels, dimension = (3,2) if self.config['dataset'].get("img") else (2,1)
        kernel_size = 3
        batch_size = x.shape[0]
        # using default kaiming's initiliation, according to the rand_conv paper
        if self.config['model']['conv_type'] == "plain":
            conv = nn.Conv2d if self.config['dataset'].get("img") else nn.Conv1d
            rand_conv_layer = conv(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size, padding="same")
        elif self.config['model']['conv_type'] == "block":
            conv = torchvision.ops.DeformConv2d if self.config['dataset'].get("img") else DeformConv1d
            padding = 1 if self.config['dataset'].get("img") else "same"
            rand_conv_layer = conv(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size, padding=padding)
        else :
            raise Exception("what kind of random  to use?")
        
        # initializa distribution parameters
        sigma_gauss_filter = np.random.uniform(1e-7,1.0)
        # initialize random weights 
        gaussian_filter = fspecial_gauss(num_channels, sigma_gauss_filter)
        gaussian_filter = torch.from_numpy(gaussian_filter).to(x)
        rand_conv_layer.to(x)
        rand_conv_layer.weight =  torch.nn.Parameter(rand_conv_layer.weight * gaussian_filter)
        
        
        if self.config['model']['conv_type'] == "plain":
            for _ in range(self.config['model']["conv_reptitions"]):
                x = rand_conv_layer(x)
        elif self.config['model']['conv_type'] == "block":   
            """   parameter initialize   """  
            # 1. deformable conv offset          
            upper_bound_distortion_sigma = 0.5 if self.config['dataset'].get("img") else 0.1
            distortion_scale = np.random.uniform(1e-7, upper_bound_distortion_sigma)
            GRF_coeff = 10
            def GRF_wrapper(any):
                return gaussian_random_field(alpha=GRF_coeff, size=224, scale=distortion_scale)

            GRF_nums = dimension * (kernel_size**dimension)
            offset_place_holder = np.zeros((batch_size, GRF_nums,1))
            offset = np.apply_along_axis(GRF_wrapper, 2, offset_place_holder)
            # offset = np.random.normal(size = (63, GRF_nums , 224, 224))
            offset = torch.from_numpy(offset).to(x)
            kwargs = {"offset":offset} if self.config['dataset'].get("img") else {}
            
            # 2. affine tranform
            sigma_affine_mean, sigma_affine_offset = 0.5, 0.5
            affine_mean = np.random.normal(0,sigma_affine_mean)
            affine_offset = np.random.normal(0,sigma_affine_offset)
            for _ in range(self.config['model']["conv_reptitions"]):
                x = rand_conv_layer(x, **kwargs) # deformable conv
                x = self.instance_norm(x)# standardize
                x = affine_mean*x + affine_offset # affine tranform
                x = self.tanh(x) # tanh
        del rand_conv_layer
        return x
    
def fspecial_gauss(size, sigma):
    #  https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g
    


def fftind(size):
    """ Returns a np array of shifted Fourier coordinates k_x k_y.
        
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
    k_ind = np.mgrid[:size, :size] - int( (size + 1)/2 )
    k_ind = scipy.fftpack.fftshift(k_ind)
    return( k_ind )

def gaussian_random_field(alpha = 10.0,
                          size = 128, 
                          scale = 1,
                          flag_normalize = True):
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
        
        # Defines momentum indices
    k_idx = fftind(size)

        # Defines the amplitude as a power law 1/|k|^(alpha/2)
    amplitude = np.power( k_idx[0]**2 + k_idx[1]**2 + 1e-10, -alpha/4.0 )
    amplitude[0,0] = 0
    
        # Draws a complex gaussian random noise with normal
        # (circular) distribution
    noise = np.random.normal(scale = scale, size = (size, size)) \
        + 1j * np.random.normal(scale = scale, size = (size, size))
    
        # To real space
    gfield = np.fft.ifft2(noise * amplitude).real
    
        # Sets the standard deviation to one
    if flag_normalize:
        gfield = gfield - np.mean(gfield)
        gfield = gfield/np.std(gfield)
        
    return gfield