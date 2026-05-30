import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from classes.blur_utils import *
from classes.utils_restoration import array2tensor, tensor2array, imread_uint, crop_center, modcrop_sf

import os
import cv2
from PIL import Image
from scipy.ndimage import zoom
import hdf5storage
from tqdm import tqdm


plt.rcParams['figure.figsize'] = (2, 2)  # Width, Height in inches


class img_PnP:

    def __init__(self, image_path, forward_model_name, forward_model_args={},
                 noise_level=0.00, kernel_path=None, color_mode='RGB',
                 crop=True, crop_size=256, seed_val=7, save_path="./results/"):
        '''
        image_path: Path to the image
        forward_model_name: Name of the forward model
        forward_model_args: Arguments for the forward model
        noise_level: Noise level to be added to the image
        kernel_path: Path to the kernel file (Custom forward model through convolution; not implemented yet)
        color_mode: 'RGB' for color images, 'L' for grayscale images
        crop: True to crop the image to the nearest multiple of scale_factor
        seed_val: Seed value for reproducibility
        save_path: Path to save the results; default is "./results/"; 
        only the method get_images() will save the images upon calling with save=True
        '''

        self.recons_status = "Not Reconstructed"
        ##### Image read #####
        self.image_path = image_path

        self.color_mode = color_mode
        self.n_channels = 3 if color_mode in ['RGB', 'ycrcb'] else 1

        if color_mode not in ['RGB', 'L','ycrcb']:
            raise ValueError(
                f"Invalid color_mode: {color_mode}. Must be one of ['RGB', 'L','ycrcb'], Use 'L' for grayscale images and 'RGB' for color images")

        if image_path.endswith(".npy"):
            self.image = np.load(image_path)
            if self.color_mode == 'L':
                self.image = np.expand_dims(self.image, axis=2)
        elif color_mode in ['RGB','ycrcb']:
            self.image = Image.open(image_path)
            self.image = self.image.convert('RGB')
            self.image = np.array(self.image)
            self.image = self.image.astype(np.float32)/255.0

        else:
            self.image = Image.open(image_path)
            self.image = self.image.convert('L')
            self.image = np.array(self.image)
            self.image = self.image.astype(np.float32)/255.0
            self.image = np.expand_dims(self.image, axis=2)
            # self.image = imread_uint(image_path, n_channels=self.n_channels)
            # self.image = self.image.astype(np.float32)/255.0

        if crop:
            self.image = crop_center(self.image, crop_size, crop_size)
            # self.image = modcrop_sf(self.image, forward_model_args.get('scale_factor', 1))
        ##### Image read done#####

        #### Set up for forward model ####

        valid_forward_models = ["deblurring", "superresolution"]

        if forward_model_name not in valid_forward_models:
            raise ValueError(
                f"Invalid forward_model: {forward_model_name}. Must be one of {valid_forward_models}")

        self.forward_model = forward_model_name
        self.forward_model_args = forward_model_args

        self.noise_level = noise_level
        self.seed_val = seed_val
        self.save_path = save_path
        if os.path.exists(self.save_path) == False:
            os.makedirs(self.save_path)
        self.init_image()
        if self.color_mode == 'ycrcb':
            self.image = cv2.cvtColor(self.image.astype(np.float32), cv2.COLOR_RGB2YCrCb)
            self.observed = cv2.cvtColor(self.observed.astype(np.float32), cv2.COLOR_RGB2YCrCb)
            self.start_image = cv2.cvtColor(self.start_image.astype(np.float32), cv2.COLOR_RGB2YCrCb)
    
    def convert_ycrcb_to_rgb(self):
        self.image = cv2.cvtColor(self.image.astype(np.float32), cv2.COLOR_YCrCb2RGB)
        self.observed = cv2.cvtColor(self.observed.astype(np.float32), cv2.COLOR_YCrCb2RGB)
        self.start_image = cv2.cvtColor(self.start_image.astype(np.float32), cv2.COLOR_YCrCb2RGB)
        if self.reconstruction is not None:
            self.reconstruction = cv2.cvtColor(self.reconstruction.astype(np.float32), cv2.COLOR_YCrCb2RGB)

    def init_image(self):

        self.set_forward_model()

        #### Set up for forward model done ####

        #### Apply forward model to generate  image ####

        self.apply_forward_model()

        np.random.seed(self.seed_val)
        self.observed = self.observed + \
            np.random.normal(0, self.noise_level, self.observed.shape)

        self.observed_tensor = torch.from_numpy(self.observed).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        
        #### Apply forward model to generate  image done ####

        #### Set start image ####
        self.set_start_image()
        self.reconstruction = None
        #### Set start image done ####

    def initialize_prox_kernel(self, img):
        '''
        calculus for future prox computatations
        :param img: degraded image
        '''

        self.FB, self.FBC, self.F2B, self.FBFy = pre_calculate_prox(img, self.kernel_tensor, self.sf)

    def data_fidelity_prox_step(self, x, step_size):
        '''
        Calculation of the proximal step on the data-fidelity term f
        '''

        y_ = x.copy()

        if x.ndim == 2:
            x = np.expand_dims(x, axis=2)

        curr_img = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self.device)

        step_size = step_size / self.op_norm ** 2

        y_ = prox_solution_L2(curr_img, self.FB, self.FBC, self.F2B, self.FBFy, step_size, self.sf)

        y_ = y_.cpu().squeeze(0).permute(1, 2, 0).detach().numpy().astype(np.float32)
        return y_

    def set_forward_model(self):
        # required args, kernel_id and scale_factor
        try:
            kernel_path = '../images/kernels/Levin09.mat'
            kernels = hdf5storage.loadmat(kernel_path)['kernels']

            self.kernel_id = self.forward_model_args['kernel_id']
            if self.kernel_id < 8 and self.kernel_id >= 0:
                self.kernel = kernels[0,
                                      self.forward_model_args['kernel_id']]

            if self.kernel_id == 8:  # Gaussian blur
                def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
                    """
                    2D gaussian mask - should give the same result as MATLAB's
                    fspecial('gaussian',[shape],[sigma])
                    """
                    m, n = [(ss-1.)/2. for ss in shape]
                    y, x = np.ogrid[-m:m+1, -n:n+1]
                    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
                    h[h < np.finfo(h.dtype).eps*h.max()] = 0
                    sumh = h.sum()
                    if sumh != 0:
                        h /= sumh
                    return h
                self.kernel = matlab_style_gauss2D(
                    shape=(25, 25), sigma=1.6)
            if self.kernel_id == 9:  # Box blur
                self.kernel = (1/81)*np.ones((9, 9))
            if self.kernel_id >= 10:
                raise ValueError(
                    "Invalid kernel_id. Must be less than 10")

            self.sf = self.forward_model_args['scale_factor']
            self.device = self.forward_model_args['device']
        except:
            raise ValueError(
                "kernel_id and scale_factor and device must be provided for prox")

        # Convert kernel to tensor
        self.kernel_tensor = torch.from_numpy(self.kernel).float().to(self.device)

        self.A_function = G
        self.A_function_adjoint = Gt
        self.A_kwargs = {'k': self.kernel_tensor, 'sf': self.sf}
        self.A_adjoint_kwargs = {'k': self.kernel_tensor, 'sf': self.sf}
        self.op_norm = 1

        for keys_input in self.forward_model_args.keys():
            for keys_default in self.A_adjoint_kwargs.keys():
                if keys_input == keys_default:
                    self.A_kwargs[keys_input] = self.forward_model_args[keys_input]
                    self.A_adjoint_kwargs[keys_input] = self.forward_model_args[keys_input]

    def apply_forward_model(self):

        # For other forward models, apply the forward operator
        img = torch.from_numpy(self.image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        self.observed = self.A_function(
            img, **self.A_kwargs).cpu().squeeze(0).permute(1, 2, 0).detach().numpy()

    def set_start_image(self):
        init = self.forward_model_args.get('init', 'observed')
        if init == 'observed':
            if self.sf == 1:
                self.start_image = self.observed.copy()
            else:
                self.start_image = zoom(self.observed, (self.sf, self.sf, 1))
        elif init == 'transpose':
            self.start_image = self.A_function_adjoint(
                self.observed_tensor, **self.A_adjoint_kwargs).cpu().squeeze(0).permute(1, 2, 0).detach().numpy()
        elif init == 'zeros':
            self.start_image = np.zeros_like(self.image)
        elif init == 'ones':
            self.start_image = np.ones_like(self.image)
        elif init == 'uniform_random':
            self.start_image = np.random.uniform(
                low=0, high=1, size=self.image.shape).astype(np.float32)
        elif init == 'normal_random':
            self.start_image = np.random.normal(
                loc=0, scale=1, size=self.image.shape).astype(np.float32)
        elif init == 'clean':
            self.start_image = self.image.copy()

        self.initialize_prox_kernel(self.observed_tensor)

    def get_metrics(self, my_image):
        if my_image.ndim == 2:
            my_image = np.expand_dims(my_image, axis=2)
        try:
            my_psnr = psnr(self.image, my_image, data_range=1.0)
            if self.color_mode == 'L':
                my_ssim = ssim(self.image[:, :, 0],
                               my_image[:, :, 0], data_range=1.0)
            if self.color_mode == 'RGB' or self.color_mode == 'ycrcb':
                my_ssim = np.mean([ssim(self.image[:, :, 0], my_image[:, :, 0], data_range=1.0), ssim(
                    self.image[:, :, 1], my_image[:, :, 1], data_range=1.0), ssim(self.image[:, :, 2], my_image[:, :, 2], data_range=1.0)])
            
        except:
            my_psnr = None
            my_ssim = None
        return my_psnr, my_ssim

    def get_images(self, plot=False, save=False):
        if plot and self.reconstruction is None:
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 3, 1)
            plt.imshow(self.image, cmap='gray', vmin=0, vmax=1)
            plt.title("Original Image")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(self.observed, cmap='gray', vmin=0, vmax=1)
            psnr_val, ssim_val = self.get_metrics(self.observed)
            if psnr_val is None:
                plt.title(f"Observed Image\nPSNR: N/A, \nSSIM: N/A")
            else:
                plt.title(
                    f"Observed Image\nPSNR: {psnr_val:.2f}, \nSSIM: {ssim_val:.4f}")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(self.start_image, cmap='gray', vmin=0, vmax=1)
            psnr_val_start, ssim_val_start = self.get_metrics(self.start_image)
            plt.title(
                f"Start Image\nPSNR: {psnr_val_start:.2f}, \nSSIM: {ssim_val_start:.4f}")
            plt.axis('off')

            plt.savefig(self.save_path + "metrics.png", bbox_inches='tight')
            plt.show()
            plt.close()
        if plot and self.reconstruction is not None:
            plt.figure(figsize=(10, 10))
            plt.subplot(1, 4, 1)
            plt.imshow(self.image, cmap='gray', vmin=0, vmax=1)
            plt.title("Original Image")
            plt.axis('off')

            plt.subplot(1, 4, 2)
            plt.imshow(self.observed, cmap='gray', vmin=0, vmax=1)
            psnr_val, ssim_val = self.get_metrics(self.observed)
            if psnr_val is None:
                plt.title(f"Observed Image\nPSNR: N/A, \nSSIM: N/A")
            else:
                plt.title(
                    f"Observed Image\nPSNR: {psnr_val:.2f}, \nSSIM: {ssim_val:.4f}")
            plt.axis('off')

            plt.subplot(1, 4, 3)
            plt.imshow(self.start_image, cmap='gray', vmin=0, vmax=1)
            psnr_val_start, ssim_val_start = self.get_metrics(self.start_image)
            plt.title(
                f"Start Image\nPSNR: {psnr_val_start:.2f}, \nSSIM: {ssim_val_start:.4f}")
            plt.axis('off')

            plt.subplot(1, 4, 4)
            plt.imshow(self.reconstruction, cmap='gray', vmin=0, vmax=1)
            psnr_val_recons, ssim_val_recons = self.get_metrics(
                self.reconstruction)
            plt.title(
                f"Reconstructed Image\nPSNR: {psnr_val_recons:.2f}, \nSSIM: {ssim_val_recons:.4f}")
            plt.axis('off')

            plt.savefig(self.save_path + "metrics.png", bbox_inches='tight')
            plt.show()
            plt.close()

        if save and self.save_path is not None:
            if self.color_mode == 'RGB':
                rgb_image = cv2.cvtColor(
                    self.image.astype(np.float32), cv2.COLOR_BGR2RGB)
                cv2.imwrite(self.save_path +
                            "original_image.png", rgb_image*255)
                rgb_observed = cv2.cvtColor(
                    self.observed.astype(np.float32), cv2.COLOR_BGR2RGB)
                cv2.imwrite(self.save_path + "observed_image.png",
                            rgb_observed*255)
                rgb_start = cv2.cvtColor(
                    self.start_image.astype(np.float32), cv2.COLOR_BGR2RGB)
                cv2.imwrite(self.save_path + "start_image.png", rgb_start*255)
                try:
                    rgb_reconstruction = cv2.cvtColor(
                        self.reconstruction.astype(np.float32), cv2.COLOR_BGR2RGB)
                    cv2.imwrite(
                        self.save_path + "reconstructed_image.png", rgb_reconstruction*255)
                except:
                    pass
                try:
                    rgb_best_psnr_recons = cv2.cvtColor(
                        self.best_psnr_recons.astype(np.float32), cv2.COLOR_BGR2RGB)
                    cv2.imwrite(
                        self.save_path + "best_psnr_reconstruction.png", rgb_best_psnr_recons*255)
                except:
                    pass
                try:
                    rgb_best_ssim_recons = cv2.cvtColor(
                        self.best_ssim_recons.astype(np.float32), cv2.COLOR_BGR2RGB)
                    cv2.imwrite(
                        self.save_path + "best_ssim_reconstruction.png", rgb_best_ssim_recons*255)
                except:
                    pass

            else:
                cv2.imwrite(self.save_path +
                            "original_image.png", self.image*255)
                cv2.imwrite(self.save_path +
                            "observed_image.png", self.observed*255)
                cv2.imwrite(self.save_path + "start_image.png",
                            self.start_image*255)
                try:
                    cv2.imwrite(
                        self.save_path + "reconstructed_image.png", self.reconstruction*255)
                except:
                    pass
                try:
                    cv2.imwrite(
                        self.save_path + "best_psnr_reconstruction.png", self.best_psnr_recons*255)
                except:
                    pass
                try:
                    cv2.imwrite(
                        self.save_path + "best_ssim_reconstruction.png", self.best_ssim_recons*255)
                except:
                    pass

        return self.image, self.observed, self.start_image, self.reconstruction

    def save_any_image(self, image, name):
        if self.color_mode == 'RGB':
            rgb_image = cv2.cvtColor(
                image.astype(np.float32), cv2.COLOR_BGR2RGB)
            cv2.imwrite(self.save_path + name, rgb_image*255)
        else:
            cv2.imwrite(self.save_path + name, image*255)

    ''' gradient operator definition'''

    def get_Gradient(self, y):
        y_tensor = torch.from_numpy(y).float().permute(
            2, 0, 1).unsqueeze(0).to(self.device)

        grad = self.A_function_adjoint(self.A_function(y_tensor, **self.A_kwargs) - self.observed_tensor, **self.A_adjoint_kwargs).cpu().squeeze(0).permute(1, 2, 0).detach().numpy().astype(np.float32)
        return grad

    ''' Definition of the gradient operator completed'''

    ''' Gradient Descent Algorithm'''

    def grad_desc_step(self, y, step_size):
        '''
        y: Current estimate
        step_size: Step size for the algorithm
        '''
        if y.ndim == 2:
            y = np.expand_dims(y, axis=2)
        step_size = step_size / self.op_norm ** 2
        y = y - step_size*self.get_Gradient(y)
        return y

    def PnP(self, denoiser, denoiser_args, denoiser_object,
            num_iterations=10, plot_graphs=False, plot_interval=100,
            save_all_iters=False, best_psnr_recons=False, best_ssim_recons=False,
            algo_params={'transpose': True, 'name': 'FBS', 'step_size': 1.9, 'clip': False}):
        '''
        denoiser: Denoiser function to be used for denoising
        denoiser_args: Arguments for the denoiser function
        denoiser_object: Denoiser object to be used for denoising
        num_iterations: Number of iterations for the PnP algorithm
        plot_graphs: use True to plot graphs of error norm and psnr
        plot_interval: Interval for plotting the graphs
        save_all_iters: use True to save all iterations ( Also set true to find best psnr or ssim reconstructions)
        best_psnr_recons: use True to find best psnr reconstruction
        best_ssim_recons: use True to find best ssim reconstruction
        If you set both best_psnr_recons and best_ssim_recons to True, reconstruction will be the one with best psnr
        algo_params: Additional parameters for the algorithm
            Required parameters:
            name: Name of the algorithm to be used
            clip: True to clip the image to [0, 1]
            step_size: Step size for the algorithm
            transpose: True if denoiser uses torch, False if denoiser uses numpy. Set False for grayscale images even if using with torch
        '''

        y_old = self.start_image.copy()
        y = self.start_image.copy()

        default_algo_params = {'transpose': True, 'name': 'FBS', 'step_size': 1.9, 'clip': False}
        # Update algo_params with defaults if not provided
        for key1, value1 in algo_params.items():
            if key1 in default_algo_params.keys():
                default_algo_params[key1] = value1
                
        # algo_params = default_algo_params.copy()



        self.algo_params = default_algo_params

        if "name" not in self.algo_params:
            raise ValueError("name not provided in algo_params")

        algo_name = self.algo_params["name"]

        denoiser_args['device'] = self.device
        

        N = []
        all_iters = [y_old]
        psnr_value, ssim_value = self.get_metrics(y_old)
        psnrs = [psnr_value]
        ssims = [ssim_value]
        Lips = []

        for i in tqdm(range(num_iterations)):

            try:
                method = getattr(self, algo_name)
            except AttributeError:
                raise ValueError(f"Invalid algorithm name: {algo_name}")
            # method = self.FBS
            with torch.no_grad():
                # Apply the algorithm
                y = method(y, denoiser, denoiser_args, denoiser_object)

            psnr_value, ssim_value = self.get_metrics(y)

            error_norm = np.linalg.norm(y.ravel()-y_old.ravel(), 2)

            N.append(error_norm)

            psnrs.append(psnr_value)
            ssims.append(ssim_value)

            if i > 0:
                Lips.append(N[i]/N[i-1])

            if plot_graphs:

                if i % plot_interval == 0 and i > 1:
                    # plot the image
                    plt.figure(figsize=(6, 1))
                    plt.imshow(y, cmap='gray')
                    plt.axis('off')
                    plt.show()
                    plt.close()

                    plt.figure(figsize=(12, 3))

                    # Plot N
                    plt.subplot(1, 3, 1)
                    plt.plot(N)
                    plt.title(r'$\|x_{k+1} - x_k\|_2$')
                    plt.xscale('symlog')
                    plt.yscale('log')

                    # Plot psnrs
                    plt.subplot(1, 3, 2)
                    plt.plot(psnrs)
                    plt.title('PSNRs')
                    plt.xscale('symlog')
                    # plt.yscale('log')

                    # Plot lower bound on the Lipschitz constant
                    plt.subplot(1, 3, 3)
                    plt.plot(Lips)
                    plt.title('Lipschitz constants\n for D o G')
                    plt.xscale('symlog')
                    # Add red dotted line at y=1
                    plt.axhline(y=1, color='r', linestyle='--')

                    plt.show()
                    plt.close()

                # if i % plot_interval == 0 and i > 1:
                    print("Iteration:", i)
                    print("PSNR:", psnr_value)
                    print("Norm:", error_norm)
                    print("-------------------------------")

            y_old = y
            if save_all_iters:
                all_iters.append(y.copy())

        self.recons_status = "Reconstructed using PnP_{} and prior {}".format(
            self.algo_params["name"], denoiser.__name__)
        self.reconstruction = y.copy()
        self.error_norms = N
        self.psnrs = psnrs
        self.ssims = ssims
        self.lips = Lips
        self.best_psnr = np.max(psnrs)
        self.best_ssim = np.max(ssims)
        if save_all_iters:
            self.all_iters = all_iters
        else:
            self.all_iters = None

        if best_psnr_recons or best_ssim_recons:
            if not save_all_iters:
                raise ValueError(
                    "save_all_iters must be True to find best psnr or ssim reconstructions")
            # if best_psnr_recons and best_ssim_recons:
                # raise ValueError("Can't find both best psnr and ssim reconstructions. Choose one")

            best_psnr = np.argmax(psnrs)
            best_ssim = np.argmax(ssims)

            if best_ssim_recons:
                self.best_ssim_recons = all_iters[best_ssim]
                self.reconstruction = all_iters[best_ssim]
            if best_psnr_recons:
                self.best_psnr_recons = all_iters[best_psnr]
                self.reconstruction = all_iters[best_psnr]

    def RED(self, y, denoiser, denoiser_args, denoiser_object):
        '''
        x : input image
        denoiser: Denoiser function to be used for denoising
        denoiser_args: Arguments for the denoiser function
        denoiser_object: Denoiser object to be used for denoising
        **algo_params: Additional parameters for the algorithm
        '''

        step_size = self.algo_params['step_size'] / self.op_norm ** 2
        transpose = self.algo_params['transpose']
        clip = self.algo_params['clip']

        # applying the gradient step
        yG = self.get_Gradient(y)
        yG = yG.astype(np.float32)

        # applying deniser
        if transpose:
            x = denoiser(y.transpose(2, 0, 1),
                         denoiser_object, **denoiser_args)
            x = x.transpose(1, 2, 0)
        else:
            x = denoiser(y, denoiser_object, **denoiser_args)

        if clip:
            x = np.clip(x, 0, 1)

        y_temp = yG + step_size*(y-x)
        lambda_ = self.algo_params.get(
            'lambda', 2/(step_size + self.op_norm ** 2))
        y = y - lambda_*y_temp

        return y.copy()

    def FBS(self, y, denoiser, denoiser_args, denoiser_object):

        step_size = self.algo_params['step_size']
        transpose = self.algo_params['transpose']
        clip = self.algo_params['clip']

        y_temp = self.grad_desc_step(y, step_size)
        y_temp = y_temp.astype(np.float32)

        # applying deniser
        if transpose:
            x = denoiser(y_temp.transpose(2, 0, 1),
                         denoiser_object, **denoiser_args)
            x = x.transpose(1, 2, 0)
        else:
            x = denoiser(y_temp, denoiser_object, **denoiser_args)

        if clip:
            x = np.clip(x, 0, 1)

        return x.copy()

    def HQS(self, y, denoiser, denoiser_args, denoiser_object):

        step_size = self.algo_params['step_size']
        transpose = self.algo_params['transpose']
        clip = self.algo_params['clip']

        if self.color_mode == 'ycrcb':
            y_temp = cv2.cvtColor(y.astype(np.float32), cv2.COLOR_YCrCb2RGB)
        else:
            y_temp = y

        y_temp = self.data_fidelity_prox_step(y_temp, step_size)
        y_temp = y_temp.astype(np.float32)

        if self.color_mode == 'ycrcb':
            y_temp = cv2.cvtColor(y_temp.astype(np.float32), cv2.COLOR_RGB2YCrCb)


        # applying denoiser
        if transpose:
            x = denoiser(y_temp.transpose(2, 0, 1),
                         denoiser_object, **denoiser_args)
            x = x.transpose(1, 2, 0)
        else:
            x = denoiser(y_temp, denoiser_object, **denoiser_args)
        

        if clip:
            x = np.clip(x, 0, 1)
        

        return x.copy()

    def DRS(self, y, denoiser, denoiser_args, denoiser_object):

        step_size = self.algo_params['step_size']
        transpose = self.algo_params['transpose']
        clip = self.algo_params['clip']

        y_temp_half = self.data_fidelity_prox_step(y, step_size)
        y_temp_half = y_temp_half.astype(np.float32)

        # applying deniser
        if transpose:
            y_temp = denoiser((2*y_temp_half-y).transpose(2, 0, 1),
                              denoiser_object, **denoiser_args)
            y_temp = y_temp.transpose(1, 2, 0)

        else:
            y_temp = denoiser(2*y_temp_half-y,
                              denoiser_object, **denoiser_args)

        if clip:
            y_temp = np.clip(y_temp, 0, 1)

        y_temp = y_temp.astype(np.float32)

        x = y + (y_temp - y_temp_half)

        return x.copy()

    def DRSdiff(self, y, denoiser, denoiser_args, denoiser_object):

        step_size = self.algo_params['step_size']
        transpose = self.algo_params['transpose']
        clip = self.algo_params['clip']

        # applying deniser
        if transpose:
            y_temp_half = denoiser(y.transpose(
                2, 0, 1), denoiser_object, **denoiser_args)
            y_temp_half = y_temp_half.transpose(1, 2, 0)
        else:
            y_temp_half = denoiser(y, denoiser_object, **denoiser_args)

        y_temp_half = y_temp_half.astype(np.float32)

        if clip:
            y_temp_half = np.clip(y_temp_half, 0, 1)

        y_temp = self.data_fidelity_prox_step(
            2*y_temp_half-y, step_size)

        y_temp = y_temp.astype(np.float32)

        x = y + (y_temp - y_temp_half)

        return x.copy()

    def RED_PG(self, y, denoiser, denoiser_args, denoiser_object):

        step_size = self.algo_params['step_size']
        transpose = self.algo_params['transpose']
        clip = self.algo_params['clip']
        L = self.algo_params['theta']

        y_temp = self.data_fidelity_prox_step(y, step_size/L)
        y_temp = y_temp.astype(np.float32)

        # applying deniser
        if transpose:
            x = denoiser(y_temp.transpose(2, 0, 1),
                         denoiser_object, **denoiser_args)
            x = x.transpose(1, 2, 0)
        else:
            x = denoiser(y_temp, denoiser_object, **denoiser_args)

        if clip:
            x = np.clip(x, 0, 1)

        y = (x + (L-1)*y_temp) / L

        return y.copy()

    def RED_PRO(self, y, denoiser, denoiser_args, denoiser_object):

        step_size = self.algo_params['step_size']
        transpose = self.algo_params['transpose']
        clip = self.algo_params['clip']
        L = self.algo_params['theta']

        # applying the gradient step
        y_temp = self.grad_desc_step(y, step_size)
        y_temp = y_temp.astype(np.float32)

        # applying deniser
        if transpose:
            x = denoiser(y_temp.transpose(2, 0, 1),
                         denoiser_object, **denoiser_args)
            x = x.transpose(1, 2, 0)
        else:
            x = denoiser(y_temp, denoiser_object, **denoiser_args)

        if clip:
            x = np.clip(x, 0, 1)

        y = L * x + (1 - L) * y_temp

        return y.copy()
