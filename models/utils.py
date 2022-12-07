import torch
import os
import torch_fidelity
import glob
import cv2
from PIL import Image
import numpy as np
#----------------------------------------------------#
# evaluation modules
#----------------------------------------------------#
class SimpleLogger:
    def __init__(self, path):
        self.path = path

    def log(self, iteration, max_iteration, metric_dict, verbose=False):
        message = '[%03d/%03d] ' % (iteration, max_iteration)
        for key in metric_dict:
            message += '\t %s:%.3f \t' % (key, metric_dict[key])
        if verbose:
            print(message)
        record = open(self.path, 'a')
        record.write('\n' + message + '\n')
        record.close()

    def log_message(self, message, verbose=True):
        record = open(self.path, 'a')
        record.write('\n' + message + '\n')
        if verbose:
            print(message)
        record.close()

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].clamp(-1.0, 1.0).cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def save_image_numpy(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio is None:
        pass
    elif aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    elif aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)

@torch.no_grad()
def eval_loader(model, test_loader_a, test_loader_b, output_directory, opt):
    if hasattr(model, 'netG_A') and hasattr(model, 'netG_B'):
        return eval_loader_two(model, test_loader_a, test_loader_b, output_directory, opt)
    return eval_loader_one(model, test_loader_a, test_loader_b, output_directory, opt)


@torch.no_grad()
def eval_loader_one(model, test_loader_a, test_loader_b, output_directory, opt):
    fake_dir = os.path.join(output_directory, 'fake')
    if not os.path.exists(fake_dir):
        os.mkdir(fake_dir)
    if opt.direction == 'AtoB':
        test_loader = test_loader_a
        real_dir = os.path.join(opt.dataroot, 'testB')
    else:
        test_loader = test_loader_b
        real_dir = os.path.join(opt.dataroot, 'testA')

    for it, data in enumerate(test_loader):
        fake = model.netG(data['A'].cuda())
        path_fake = os.path.join(fake_dir, os.path.basename(data['A_paths'][0]).replace('jpg', 'png'))
        im = tensor2im(fake)
        save_image_numpy(im, path_fake)

    eval_dict = eval_method_one(real_dir, fake_dir, opt)
    return eval_dict

@torch.no_grad()
def eval_method_one(realB_path, fakeB_path, opt):
    print('[*] start evaluation!')
    print(realB_path)
    print(fakeB_path)
    eval_dict = {}
    if 'maps' in opt.dataroot and opt.direction == 'AtoB':
        eval_dict = eval_maps(realB_path, fakeB_path, name='A2B')
    elif 'maps' in opt.dataroot and opt.direction == 'BtoA':
        eval_dict = eval_maps(realB_path, fakeB_path, name='BtoA', thr1=10, thr2=30)
    elif 'facades' in opt.dataroot and opt.direction == 'AtoB':
        eval_dict = eval_maps(realB_path, fakeB_path, name='A2B')
    elif ('city' in opt.dataroot and opt.direction == 'AtoB'):
        eval_dict = eval_city2parsing(realB_path, fakeB_path)
    eval_args = {'fid': True, 'kid': True, 'kid_subset_size': 50, 'kid_subsets': 10, 'verbose': False, 'cuda': True}
    metric_dict_AB = torch_fidelity.calculate_metrics(input1=realB_path, input2=fakeB_path, **eval_args)
    eval_dict['FID'] = metric_dict_AB['frechet_inception_distance']
    eval_dict['KID'] = metric_dict_AB['kernel_inception_distance_mean']*100.
    print('[*] evaluation finished!')
    return eval_dict

@torch.no_grad()
def eval_loader_two(model, test_loader_a, test_loader_b, output_directory, opt):
    fakeA_dir = os.path.join(output_directory, 'fakeA')
    fakeB_dir = os.path.join(output_directory, 'fakeB')
    if not os.path.exists(fakeA_dir):
        os.mkdir(fakeA_dir)
    if not os.path.exists(fakeB_dir):
        os.mkdir(fakeB_dir)

    for it, data in enumerate(test_loader_a):
        fake = model.netG_A(data['A'].cuda())
        path_fake = os.path.join(fakeB_dir, os.path.basename(data['A_paths'][0]).replace('jpg', 'png'))
        im = tensor2im(fake)
        save_image_numpy(im, path_fake)

    for it, data in enumerate(test_loader_b):
        fake = model.netG_B(data['A'].cuda())
        path_fake = os.path.join(fakeA_dir, os.path.basename(data['A_paths'][0]).replace('jpg', 'png'))
        im = tensor2im(fake)
        save_image_numpy(im, path_fake)

    realA_dir, realB_dir = os.path.join(opt.dataroot, 'testA'), os.path.join(opt.dataroot, 'testB')
    eval_dict = eval_method_two(realA_dir, realB_dir, fakeA_dir, fakeB_dir, opt)
    return eval_dict


@torch.no_grad()
def eval_method_two(realA_path, realB_path, fakeA_path, fakeB_path, opt):
    eval_dict = {}
    if 'maps' in opt.dataroot:
        eval_dict = eval_maps(realB_path, fakeB_path, name='A2B')
    elif 'city' in opt.dataroot:
        eval_dict = eval_city2parsing(realB_path, fakeB_path)
    eval_args = {'fid': True, 'kid': True, 'kid_subset_size': 50, 'kid_subsets': 10, 'verbose': False, 'cuda': True}
    metric_dict_AB = torch_fidelity.calculate_metrics(input1=realB_path, input2=fakeB_path, **eval_args)
    metric_dict_BA = torch_fidelity.calculate_metrics(input1=realA_path, input2=fakeA_path, **eval_args)
    eval_dict['FID_A2B'] = metric_dict_AB['frechet_inception_distance']
    eval_dict['KID_A2B'] = metric_dict_AB['kernel_inception_distance_mean']*100.
    eval_dict['FID_B2A'] = metric_dict_BA['frechet_inception_distance']
    eval_dict['KID_B2A'] = metric_dict_BA['kernel_inception_distance_mean']*100.
    return eval_dict

def eval_maps(real_path, fake_path, thr1=5, thr2=10, name=''):
    reals = glob.glob(real_path + '/*')
    fakes = glob.glob(fake_path + '/*')

    reals = sorted(reals)
    fakes = sorted(fakes)
    print(real_path, fake_path)

    num_imgs = len(reals)
    corr5_count = 0.0
    corr10_count = 0.0
    pix_count = 0.0
    RMSE = 0.0
    for i in range(num_imgs):

        real = cv2.imread(reals[i])
        fake = cv2.imread(fakes[i])

        real = cv2.resize(real, (256, 256), interpolation=cv2.INTER_LINEAR)
        fake = cv2.resize(fake, (256, 256), interpolation=cv2.INTER_LINEAR)

        real = real.astype(np.float32)
        fake = fake.astype(np.float32)
        diff = np.abs(real - fake)

        max_diff = np.max(diff, axis=2)

        corr5_count = corr5_count + np.sum(max_diff < thr1)
        corr10_count = corr10_count + np.sum(max_diff < thr2)
        pix_count = pix_count + 256**2

        diff = (diff**2)/(256**2)
        diff = np.sum(diff)
        rmse = np.sqrt(diff)
        RMSE = RMSE + rmse

    RMSE = RMSE/num_imgs
    acc5 = corr5_count/pix_count*100.
    acc10 = corr10_count/pix_count*100.
    eval_dict = {'%s/rmse' % (name):RMSE,'%s/acc@%d'%(name, thr1):acc5, '%s/acc@%d'%(name, thr2):acc10}
    return eval_dict



def eval_city2parsing(real_path, fake_path):
    labels = [{'name':'road', 'catId':0, 'color': (128, 64, 128)},
              {'name':'sidewalk', 'catId':1, 'color': (244, 35, 232)},
              {'name':'building', 'catId':2, 'color': (70, 70, 70)},
              {'name':'wall', 'catId':3, 'color': (102, 102, 156)},
              {'name':'fence', 'catId':4, 'color': (190, 153, 153)},
              {'name':'pole', 'catId':5, 'color': (153, 153, 153)},
              {'name':'traffic_light', 'catId':6, 'color': (250, 170, 30)},
              {'name':'traffic_sign', 'catId':7, 'color': (220, 220, 0)},
              {'name':'vegetation', 'catId':8, 'color': (107, 142, 35)},
              {'name':'terrain', 'catId':9, 'color': (152, 251, 152)},
              {'name':'sky', 'catId':10, 'color': (70, 130, 180)},
              {'name':'person', 'catId':11, 'color': (220, 20, 60)},
              {'name':'rider', 'catId':12, 'color': (255, 0, 0)},
              {'name':'car', 'catId':13, 'color': (0, 0, 142)},
              {'name':'truck', 'catId':14, 'color': (0, 0, 70)},
              {'name':'bus', 'catId':15, 'color': (0, 60, 100)},
              {'name':'train', 'catId':16, 'color': (0, 80, 100)},
              {'name':'motorcycle', 'catId':17, 'color': (0, 0, 230)},
              {'name':'bicycle', 'catId':18, 'color': (119, 11, 32)},
              {'name':'ignore', 'catId':19, 'color': (0, 0, 0)}]

    reals = glob.glob(real_path+'/*jpg')
    fakes = glob.glob(fake_path+'/*png')
    reals = sorted(reals)
    fakes = sorted(fakes)
    num_imgs = len(reals)

    CM = np.zeros((19,19), dtype=np.float32)
    # test
    for i in range(num_imgs):
        real = cv2.imread(reals[i])
        fake = cv2.imread(fakes[i])

        real = cv2.resize(real, (128, 128), interpolation=cv2.INTER_NEAREST)
        fake = cv2.resize(fake, (128, 128), interpolation=cv2.INTER_NEAREST)

        pred = fake
        label = real


        label_dis = np.zeros((20, 128, 128), dtype=np.float32)
        pred_dis = np.zeros((20, 128, 128), dtype=np.float32)

        for j in range(20):
            color = labels[j]['color']
            label_diff = np.abs(label - color)
            pred_diff = np.abs(pred - color)

            label_diff = np.sum(label_diff, axis=2)
            pred_diff = np.sum(pred_diff, axis=2)

            label_dis[j,:,:] = label_diff
            pred_dis[j,:,:] = pred_diff

        label_id = np.argmin(label_dis, axis=0)
        pred_id = np.argmin(pred_dis, axis=0)

        for j in range(19):
            coord = np.where(label_id == j)
            pred_j = pred_id[coord]
            for k in range(19):
                CM[j,k] = CM[j, k] + np.sum(pred_j == k)


    pix_acc = 0
    mean_acc = 0
    mean_IoU = 0

    count = 0
    for i in range(19):
        count = count + CM[i, i]
    pix_acc = count / np.sum(CM)

    count = 0
    for i in range(19):
        temp = CM[i, :]
        count = count + CM[i,i]/(np.sum(temp) + 1e-6)
    mean_acc = count/19

    count = 0
    for i in range(19):
        temp_0 = CM[i, :]
        temp_1 = CM[:, i]
        count = count + CM[i, i]/(np.sum(temp_0) + np.sum(temp_1) - CM[i, i] + 1e-6)

    mean_IoU = count/19

    eval_dict = {'pix_acc':pix_acc, 'mean_acc':mean_acc, 'mean_IoU':mean_IoU}
    return eval_dict


