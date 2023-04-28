import os
from reader import sintel, kitti
import cv2
import numpy as np
import skimage.io
import torch
from VP.utilities import gpu_reverse_flow, jacobian_det, reverse_flow
from VP.vis_utilities import array_hist_image

# PLEASE MODIFY the paths specified in sintel.py and kitti.py
def predict_sintel_kitti(pipe, prefix, batch_size = 8, resize = None):

    sintel_resize = (448, 1024) if resize is None else resize
    sintel_dataset = sintel.list_data()
    prefix = prefix + '_sintel'
    if not os.path.exists(prefix):
        os.mkdir(prefix)

    flo = sintel.Flo(1024, 436)

    for div in ('test',):
        for k, dataset in sintel_dataset[div].items():
            if k == 'clean':
                continue
            output_folder = os.path.join(prefix, k)
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)
            img1, img2 = [[sintel.load(p) for p in data] for data in list(zip(*dataset))[:2]]
            for result, entry in zip(pipe.predict(img1, img2, batch_size = 1, resize = sintel_resize), dataset):
                flow, occ_mask, warped = result
                img1 = entry[0]
                fname = os.path.basename(img1)
                seq = os.path.basename(os.path.dirname(img1))
                seq_output_folder = os.path.join(output_folder, seq)
                if not os.path.exists(seq_output_folder):
                    os.mkdir(seq_output_folder)
                flo.save(flow, os.path.join(seq_output_folder, fname.replace('.png', '.flo')))
                skimage.io.imsave(os.path.join(seq_output_folder, fname), np.clip(warped * 255, 0, 255).astype(np.uint8))

    '''
	KITTI 2012:
	Submission instructions: For the optical flow benchmark, all flow fields of the test set must be provided in the root directory of a zip file using the file format described in the readme.txt (16 bit color png) and the file name convention of the ground truth (000000_10.png, ... , 000194_10.png).

	KITTI 2015:
	Submission instructions: Provide a zip file which contains the 'disp_0' directory (stereo), the 'flow' directory (flow), or the 'disp_0', 'disp_1' and 'flow' directories (scene flow) in its root folder. Use the file format and naming described in the readme.txt (000000_10.png,...,000199_10.png).
	'''

    kitti_resize = (512, 1152) if resize is None else resize
    kitti_dataset = kitti.read_dataset_testing(resize = kitti_resize)
    prefix = prefix.replace('sintel', 'kitti')
    if not os.path.exists(prefix):
        os.mkdir(prefix)

    for k, dataset in kitti_dataset.items():
        output_folder = os.path.join(prefix, k)
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        img1 = kitti_dataset[k]['image_0']
        img2 = kitti_dataset[k]['image_1']
        cnt = 0
        for result in pipe.predict(img1, img2, batch_size = 1, resize = kitti_resize):
            flow, occ_mask, warped = result
            out_name = os.path.join(output_folder, '%06d_10.png' % cnt)
            cnt = cnt + 1

            pred = np.ones((flow.shape[0], flow.shape[1], 3)).astype(np.uint16)
            pred[:, :, 2] = (64.0 * (flow[:, :, 0] + 512)).astype(np.uint16)
            pred[:, :, 1] = (64.0 * (flow[:, :, 1] + 512)).astype(np.uint16)
            cv2.imwrite(out_name, pred)

def add_landmarks(img, lmk, color = (0, 0, 255)):
    img = img.copy()
    lmk = lmk.astype(np.int32)
    for i in range(lmk.shape[0]):
        x, y = lmk[i]
        cv2.circle(img, (y, x), 3, color, -1)
    return img

def predict(pipe, dataset, save_dir, batch_size=8, resize = None):
    resize = (512, 512) if resize is None else resize
    prefix = save_dir
    if not os.path.exists(prefix):
        # make parent directory if not exist
        import pathlib
        pathlib.Path(prefix).mkdir(parents=True, exist_ok=True)
    print('Save to {}'.format(prefix))

    cnt = 0
    for i in range(0, len(dataset), batch_size):
        this_batch_size = min(batch_size, len(dataset) - i)

        print("predicting on {} to {}".format(i, i + this_batch_size), end="\r", flush=True)
        img1 = [dataset[k]['image_0'] for k in range(i, i + this_batch_size)]
        img2 = [dataset[k]['image_1'] for k in range(i, i + this_batch_size)]
        fids = [dataset[k]['fid'] for k in range(i, i + this_batch_size)]

        for fid, result in zip(fids, pipe.predict(img1, img2, batch_size = 1, resize = resize)):
            output_folder = os.path.join(prefix, fid)
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)
            flow, occ_mask, warped = result
            # lmk_warped = pipe.reconstruction()

            pred = np.ones((flow.shape[0], flow.shape[1], 3)).astype(np.uint16)
            pred[:, :, 2] = (64.0 * (flow[:, :, 0] + 512)).astype(np.uint16)
            pred[:, :, 1] = (64.0 * (flow[:, :, 1] + 512)).astype(np.uint16)

            img1_lmk = add_landmarks(img1[0], dataset[i]['lmk_0'], color=(0, 255, 0))
            img2_lmk = add_landmarks(img2[0], dataset[i]['lmk_1'], color=(0, 0, 255))
            warped = warped * 255
            warped_lmk = add_landmarks(warped, dataset[i]['lmk_0'], color=(0, 255, 0))
            warped_lmk = add_landmarks(warped_lmk, dataset[i]['lmk_1'], color=(0, 0, 255))
            save_dict = {
                os.path.join(output_folder, f'{fid}_flow.png'): pred,
                os.path.join(output_folder, f'{fid}_warped.png'): warped_lmk,
                os.path.join(output_folder, f'{fid}_img0.png'): img1_lmk,
                os.path.join(output_folder, f'{fid}_img1.png'): img2_lmk,
            }
            cnt = cnt + 1

            for k,v in save_dict.items():
                cv2.imwrite(k, v)

def visualize(pipe, dataset, save_dir='/home/hynx/regis/SFG/tmp', resize = None):
    resize = (512, 512) if resize is None else resize

    if save_dir:
        prefix = save_dir
        if not os.path.exists(prefix):
            # make parent directory if not exist
            import pathlib
            pathlib.Path(prefix).mkdir(parents=True, exist_ok=True)
        print('Save to {}'.format(prefix))

    cnt = 0

    img1 = [dataset[k]['image_0'] for k in range(len(dataset))]
    img2 = [dataset[k]['image_1'] for k in range(len(dataset))]
    fids = [dataset[k]['fid'] for k in range(len(dataset))]

    for fid, result in zip(fids, pipe.predict(img1, img2, batch_size = 8, resize = resize)):
        flow, occ_mask, warped = result
        print('processing {}'.format(fid))

        # to torch tensor
        flow = flow.transpose(2,0,1)
        rev_flow = reverse_flow(flow)
        rev_flow = torch.from_numpy(rev_flow)
        jac = vis_step(rev_flow).squeeze().numpy()
        # nomralize to 0-255
        jac = (jac - jac.min()) / (jac.max() - jac.min()) * 255
        jac_hist = array_hist_image(jac)
        if save_dir:
            output_folder = os.path.join(prefix, fid)
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)
            save_dict = {
                os.path.join(output_folder, f'{fid}_jac.png'): jac,
                os.path.join(output_folder, f'{fid}_warped.png'): warped * 255,
                os.path.join(output_folder, f'{fid}_fixed.png'): img1[cnt],
                os.path.join(output_folder, f'{fid}_moving.png'): img2[cnt],
                os.path.join(output_folder, f'{fid}_jac_hist.png'): jac_hist,
            }

            for k,v in save_dict.items():
                cv2.imwrite(k, v)
            import pdb; pdb.set_trace()

        cnt = cnt + 1

def vis_step(flow):
    jac = jacobian_det(flow)

    return jac