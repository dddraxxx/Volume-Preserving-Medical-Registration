# validate the MaskFlownet_S model on ANHIR dataset.
# Load the pretrained model from 2afApr28
# only do valid
python train_baseline.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 0 -c 2afApr28 --batch 1 --valid

# do predict
python train_baseline.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 0 -c 2afApr28 --batch 8 --predict_fold train --predict

# do visualize
python train_baseline.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 0 -c 2afApr28 --relative UM --visualize

# do training
python train_baseline.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 0 --clear_steps --weight 200 --batch 8 --relative UM