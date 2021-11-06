import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from itertools import islice
from util import html
import pdb
import lpips

# options
opt = TestOptions().parse()
opt.num_threads = 1   # test code only supports num_threads=1
opt.batch_size = 1   # test code only supports batch_size=1
opt.serial_batches = True  # no shuffle

# dataset: discover60k
opt.discover60k_kwargs = {
    'input_dir': '/home/jshi31/dataset/discover60k/before',
    'output_dir': '/home/jshi31/dataset/discover60k/after',
    'anno_path': '/home/jshi31/dataset/discover60k/annotation/discover60k.json',
}
opt.fivek_kwargs = {            
    'anno_dir': '/home/jshi31/dataset/FiveK/annotations',
    'img_dir': '/home/jshi31/dataset/FiveK/images'
}

# create dataset
dataset = create_dataset(opt)
model = create_model(opt)
model.setup(opt)
model.eval()
print('Loading model %s' % opt.model)

# create website
web_dir = os.path.join(opt.results_dir, opt.phase + 'ep{}'.format(opt.epoch) + '_sync' if opt.sync else opt.phase)
webpage = html.HTML(web_dir, 'Training = %s, Phase = %s, Class =%s' % (opt.name, opt.phase, opt.name))

# sample random z
if opt.sync:
    z_samples = model.get_z_random(opt.n_samples + 1, opt.nz)

# Init the LPIPS
lpips_fn = lpips.LPIPS(net='alex', version='0.1').to('cuda')
lpips_score = 0


# test stage
#for i, data in enumerate(islice(dataset, opt.num_test)):
for i, data in enumerate(dataset):
    itr = i + 1
    print(data['A_paths'])
    model.set_input(data)
    print('process input image %3.3d/%3.3d' % (i, opt.num_test))
    if not opt.sync:
        z_samples = model.get_z_random(opt.n_samples + 1, opt.nz)
    gen_imgs = []
    for nn in range(opt.n_samples + 1):
        encode = nn == 0 and not opt.no_encode
        real_A, fake_B, real_B = model.test(z_samples[[nn]], encode=encode)
        if nn == 0:
            images = [real_A, real_B, fake_B]
            names = ['input', 'ground truth', 'encoded']
        else:
            images.append(fake_B)
            names.append('random_sample%2.2d' % nn)
            gen_imgs.append(fake_B)

    # Compute LPIPS
    dist_sum = 0
    for i in range(len(gen_imgs) - 1):
        for j in range(i + 1, len(gen_imgs)):
            dist = lpips_fn.forward(gen_imgs[i], gen_imgs[j]).detach()  # scale [-1 ,1]
            dist_sum += dist
    dist_avg = (dist_sum / (len(gen_imgs) * (len(gen_imgs) - 1) / 2)).reshape(1, 1)
    lpips_score = lpips_score * (1 - 1/itr) + dist_avg/itr
    if itr % 100 == 0:
        print('{}/{} lpips: {:.4f}'.format(itr, len(dataset), lpips_score))

    img_path = 'input_%3.3d' % i
    if opt.viz:
        save_images(webpage, images, names, img_path, aspect_ratio=opt.aspect_ratio, width=opt.crop_size)
print('lpips {:.4f}'.format(lpips_score))
if opt.viz:
    webpage.save()
