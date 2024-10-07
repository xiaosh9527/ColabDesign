#!/bin/bash
#SBATCH --partition h100
#SBATCH --gres gpu:1
#SBATCH --mem 16g
#SBATCH --time 24:00:00
#SBATCH --array=1-5
#SBATCH --job-name=afdesign
#SBATCH --output=./exelogs/afdesign.%A_%a.out
#SBATCH --error=./exelogs/afdesign.%A_%a.out

module load gcc/13.2.0
module load cuda/12.4.1
module load cudnn/9.2.1.18-12

echo Setting up virtual python environment...
source /home/shxiao/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate /home/shxiao/anaconda3/envs/colab_kuma
echo Colabdesign environment succesfully loaded!

SEED=$SLURM_ARRAY_JOB_ID\_$SLURM_ARRAY_TASK_ID

python -u afdesign_diffused_mob_with_mpnn_prediction_ver_2.py \
        --seed=$SEED \
        --num_diffusion_design=20 \
        --num_afdesign_designs=10 \
        --num_mpnn_designs=10 \
        --learning_rate=0.02 \
        --target_pdb=./input_pdbs/6m0j.pdb \
        --target_chain=E \
        --target_hotspot=E453,E489,E505 \
        --binder_pdb=./input_pdbs/1ttg.pdb \
        --binder_chain=A \
        --fix_pos=1-24 \
        --rm_aa=C \
        --use_multimer \
        --target_flexible \
        --use_binder_template \
        --rm_template_ic \
        --use_mpnn_loss \
        --mpnn_weights soluble \
        --mpnn_model_name v_48_010

# python -u afdesign_diffused_mob_with_mpnn_prediction_ver_2.py \
#         --seed=$SEED \
#         --num_diffusion_design=20 \
#         --num_afdesign_designs=20 \
#         --num_mpnn_designs=10 \
#         --learning_rate=0.02 \
#         --target_pdb=./input_pdbs/S2-VHH_refine_6.pdb \
#         --target_chain=D,E,F \
#         --target_hotspot=D40,D43,D47,D50,E38,E41,E45,E49 \
#         --binder_pdb=./input_pdbs/1ttg.pdb \
#         --binder_chain=A \
#         --fix_pos=1-24 \
#         --rm_aa=C \
#         --use_multimer \
#         --use_binder_template \
#         --target_flexible \
#         --rm_template_ic \
#         --use_mpnn_loss \
#         --mpnn_weights soluble \
#         --mpnn_model_name v_48_010

# python -u afdesign_diffused_mob_with_mpnn_prediction_ver_2.py \
#         --seed=$SEED \
#         --num_diffusion_design=20 \
#         --num_afdesign_designs=20 \
#         --num_mpnn_designs=10 \
#         --learning_rate=0.05 \
#         --target_pdb=./input_pdbs/5jhl_trunct.pdb \
#         --target_chain=A \
#         --target_hotspot=A101,A107,A108 \
#         --binder_pdb=./input_pdbs/1ttg.pdb \
#         --binder_chain=A \
#         --fix_pos=1-24 \
#         --rm_aa=C \
#         --use_multimer \
#         --target_flexible \
#         --rm_template_ic \
#         --use_binder_template \
#         --use_mpnn_loss \
#         --mpnn_weights soluble \
#         --mpnn_model_name v_48_010