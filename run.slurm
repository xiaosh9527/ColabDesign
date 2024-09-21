#!/bin/bash
#SBATCH --partition h100
#SBATCH --gres gpu:1
#SBATCH --mem 16g
#SBATCH --time 24:00:00
#SBATCH --array=1-10
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

python -u afdesign_diffused_mob_with_mpnn_prediction.py \
        --seed=$SEED \
        --em_map_path=./examples/7lo8_emd_23464.mrc \
        --density_cutoff=0.25 \
        --num_afdesign_designs=20 \
        --num_mpnn_designs=10 \
        --learning_rate=0.02 \
        --target_pdb=./examples/7lo8_Z.pdb \
        --target_chain=Z \
        --target_hotspot=227 \
        --binder_pdb=./examples/7lo8_HL.pdb \
        --binder_chain=H,L \
        --rm_aa=C \
        --target_flexible \
        --rm_template_ic \
        --use_binder_template \
        --use_mpnn_loss \
        --use_multimer \
        --mpnn_weights abmpnn \
        --mpnn_model_name abmpnn
