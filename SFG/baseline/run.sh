# validate the MaskFlownet_S model on ANHIR dataset.
# Load the pretrained model from 2afApr28
# only do valid
python train_baseline.py MaskFlownet_S.yaml --dataset_cfg ANHIR.yaml -g 0 -c 2afApr28 --clear_steps --weight 200 --batch 1 --relative UM --prep 512 --valid

# do 