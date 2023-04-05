import itk
import numpy as np
from skimage import io

def itk_affine(arr1, arr2):
    # arr dtype float
    # Convert input arrays to ITK image views
    fixed_image = itk.GetImageViewFromArray(np.array(arr1).astype(np.float32))
    moving_image = itk.GetImageViewFromArray(np.array(arr2).astype(np.float32))
    
    # Create parameter object and set default affine parameter map
    parameter_object = itk.ParameterObject.New()
    affine_parameter_map = parameter_object.GetDefaultParameterMap('affine')
    parameter_object.AddParameterMap(affine_parameter_map)

    # Call registration function
    result_image, result_transform_parameters = itk.elastix_registration_method(
        fixed_image, moving_image, parameter_object=parameter_object)
    res = itk.GetArrayFromImage(result_image)
    # normalize to 0-255
    res = (res - res.min())/(res.max()-res.min())*255
    return res, result_transform_parameters

def get_pair_affine(arr1, arr2):
    # Get the affine transformation parameters
    arr2r, params = itk_affine(arr1, arr2)
    tx = params.GetParameterMap(0)
    params = np.array(tx['TransformParameters'], dtype=np.float32).reshape(3,2)
    W, b = params[:2], params[-1][:2]
    o = np.array(tx['CenterOfRotationPoint'], dtype=np.float32)[:2]

    def get_total_matrix(W, b, o):
        # Create a centering matrix
        center = np.eye(3)
        center[:2, 2] = -o
        center_inv = center.copy()
        center_inv[:2, 2] = o
        # Reverse the order of the rows and columns of W
        W = (W.T)[::-1,::-1].T
        b = b[::-1]
        # Combine W and b into a single matrix
        T = np.append(W, b[None].T, axis=1)
        T = np.append(T, [[0, 0, 1]], axis=0)
        # Combine the centering matrix, T, and its inverse to get the total matrix
        return center_inv@T@center
    total_matrix = get_total_matrix(W, b, o)
    return total_matrix, arr2r


def affine_points(total_matrix, points):
    tW, tb = total_matrix[:2,:2], total_matrix[:2,2] # tW: [wx, wy, wz]
    return (points @ tW.T) + tb


if __name__=='__main__':
    save_path = '/home/hynx/regis/SFG/dataset/1024_after-affine'
    # create dir if not exist
    from pathlib import Path as pa
    pa(save_path).mkdir(parents=True, exist_ok=True)
    
    read_path = '/home/hynx/regis/SFG/dataset/1024_data_with_evaluation6_for_affine_network'
    csv_file = '/home/hynx/regis/SFG/dataset/matrix_sequence_manual_validation.csv'
    import csv
    f = csv.reader(open(csv_file, 'r'))
    
    for row in f:
        if row[0] == 'id':
            continue
        id = row[0]
        img1 = f'{read_path}/{id}_1.jpg'
        img2 = f'{read_path}/{id}_2.jpg'
        print('Processing: ', f'{id}_1.jpg', f'{id}_2.jpg')
        
        arr1 = io.imread(img1)
        arr1 = np.array(arr1, dtype=np.float32)
        arr2 = io.imread(img2)
        arr2 = np.array(arr2, dtype=np.float32)
        print(arr1.shape, arr2.shape)
        mtrix, arr1r = get_pair_affine(arr2, arr1)
        
        # save arr1r
        io.imsave(f'{save_path}/{id}_1.jpg', arr1r.astype(np.uint8)) 
        break