import os, re, sys, string, argparse, jax, subprocess, tempfile, Bio, shutil, mrcfile
sys.path.insert(0, '/work/lpdi/users/shxiao/ColabDesign/')

from utils.design import *
# from utils.align import *
from utils.io import *

import numpy as np
import pandas as pd
import jax.numpy as jnp

from scipy.special import softmax
from scipy.spatial import cKDTree 
from scipy.spatial.distance import cdist, hamming

from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.mpnn import mk_mpnn_model
from colabdesign.shared.prep import prep_pos

from Bio.PDB import *
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.Selection import unfold_entities
from Bio import SeqUtils
from Bio.Align import substitution_matrices

from abnumber import Chain

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

aligner = Bio.Align.PairwiseAligner()
aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")

AA3to1_covert = SeqUtils.IUPACData.protein_letters_3to1
AA3to1_covert = {k.upper(): v for k,v in AA3to1_covert.items()}
af_alphabet = 'ARNDCQEGHILKMFPSTWYVX'

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=str)
    parser.add_argument('--em_map_path', type=str)
    parser.add_argument('--density_cutoff', type=float, default=0.010)
    parser.add_argument('--num_afdesign_designs', type=int, default=10)
    parser.add_argument('--num_mpnn_designs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=2e-2)
    parser.add_argument('--target_pdb', type=str)
    parser.add_argument('--target_chain', type=str)
    parser.add_argument('--target_hotspot', type=str)
    parser.add_argument('--target_flexible', action='store_true', help='Whether to allow target to be flexible')
    parser.add_argument('--binder_pdb', type=str, default="1ttg")
    parser.add_argument('--binder_chain', type=str, default='A')
    parser.add_argument('--fix_pos', type=str, default=None)
    parser.add_argument('--rm_aa', type=str, default='C')
    parser.add_argument('--use_multimer', action='store_true', help='Whether to use AFmultimer')
    parser.add_argument('--num_recycles', type=int, default=1, help='Number of recycles')
    parser.add_argument('--num_models', type=str, default=2, help='Number of trained models to use during optimization; "all" for all models')
    parser.add_argument('--use_binder_template', action='store_true', help='Whether to use binder template')
    parser.add_argument('--rm_template_ic', action='store_true', help='Whether to remove template interchain information')
    parser.add_argument('--use_mpnn_loss', action='store_true', help='Whether to use MPNN loss')
    parser.add_argument('--mpnn_weights', type=str, default='abmpnn, soluble, original')
    parser.add_argument('--mpnn_model_name', type=str, default='abmpnn; v_48_002; v_48_010; v_48_020; v_48_030')
    
    return parser

def main(args) -> None:
    if args.binder_chain == "": 
        binder_chain = None
    else:
        binder_chain = args.binder_chain

    if args.use_mpnn_loss:
        print(f'\nUsing {args.mpnn_weights} weight {args.mpnn_model_name} model...\n')
    else:
        print('\nNot using MPNN loss...\n')

    seed = sum([int(x) for x in args.seed.split('_')])

    complex_name = f'{os.path.basename(args.target_pdb).split(".")[0]}_{os.path.basename(args.binder_pdb).split(".")[0]}'
    output_dir = f"./outputs/{complex_name}/job_{seed}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f'\nUsing original binder {args.binder_pdb}...\n')
    os.makedirs(output_dir+'/00.original_outputs', exist_ok=True)
    binder_pdb_prefix = output_dir+'/00.original_outputs/'+os.path.basename(args.binder_pdb).split('.pdb')[0]+'_'+str(seed)
    shutil.copy(args.binder_pdb, binder_pdb_prefix+'_000.pdb')
    iter_num = 1

    rng = np.random.default_rng(seed)
    seed_list = rng.integers(0, 2**32, args.num_afdesign_designs)
    result = {}
    for n in range(iter_num):
        binder_pdb = f'{binder_pdb_prefix}_{str(n).zfill(3)}.pdb'
        
        ''' Combine binder & target PDBs'''
        os.makedirs(os.path.join(output_dir, '01.combined_complex'), exist_ok=True)
        output_pdb_path = os.path.join(output_dir, '01.combined_complex', f'{complex_name}_{n}.pdb')
        output = combine_pdbs(
            args.target_pdb, args.target_chain,
            binder_pdb, binder_chain, 
            '/tmp/tmp.pdb',
            output_pdb_path,
            seed=seed,
            mpnn_fixed_regions=None,
            mpnn_model_name=args.mpnn_model_name, mpnn_weights=args.mpnn_weights
        )

        loop_biasing_mask = np.zeros(len(output['binder_starting_seq'].replace('/','')), dtype=bool)

        counter = 0
        for seq in output['binder_starting_seq'].split('/'):
            abnumber_chain = Chain(seq, scheme='chothia')    
            cdr1 = re.search(abnumber_chain.cdr1_seq, seq)
            cdr2 = re.search(abnumber_chain.cdr2_seq, seq)
            cdr3 = re.search(abnumber_chain.cdr3_seq, seq)
            loop_biasing_mask[counter+np.arange(*cdr1.span())] = 1.0
            loop_biasing_mask[counter+np.arange(*cdr2.span())] = 1.0
            loop_biasing_mask[counter+np.arange(*cdr3.span())] = 1.0
            counter += len(seq)
            print(len(seq))

        new_pdb = prep_pdb('/tmp/tmp.pdb', output['binder_chain'], ignore_missing=True)
        all_loop_region_residues = [f"{new_pdb['idx']['chain'][x]}{new_pdb['idx']['residue'][x]}" for x in np.where(loop_biasing_mask)[0]]
        print(f'\nStarting design with starting sequence {output["binder_starting_seq"]}.\n')
        
        if args.target_hotspot == "":                 
            target_hotspot = None
        else:
            target_hotspot = args.target_hotspot
            
        x = {
            "pdb_filename": output_pdb_path,
            "map_path": args.em_map_path,
            "density_cutoff": args.density_cutoff,
            "target_chain": output['target_chain'],
            "binder_chain": output['binder_chain'],
            "hotspot":target_hotspot,
            "use_multimer":args.use_multimer,
            "rm_target_seq":args.target_flexible,               # NOTE: make target flexible
            "use_binder_template":args.use_binder_template,     # NOTE: templating the binder
            "rm_template_ic":args.rm_template_ic,               # NOTE: remove interchain information from template
        }

        for m in range(args.num_afdesign_designs):
                                                  #seed id   #RFdiffusion     #AFdesign        #MPNN
            design_name = '_'.join([complex_name, str(seed), str(n).zfill(3), str(m).zfill(3), str(0).zfill(3)])
            result[design_name] = {"seed": seed_list[m]}

            os.makedirs(os.path.join(output_dir, '02.design_outputs'), exist_ok=True)
            design_model = design(
                x, design_name, os.path.join(output_dir, '02.design_outputs'), output['binder_starting_seq'].replace('/',''), 
                rm_aa=args.rm_aa, biasing_aa=None, biasing_mask=loop_biasing_mask, seed=seed_list[m],
                lr=args.learning_rate
            )

            aux = design_model.aux["all"]
            best_idx = aux['plddt'].mean(1).argmax()
            design_score_dict = {
                "seq":   design_model.get_seqs()[0],
                "seqid": design_model.aux['log']['seqid'],
                "loop_resid": ','.join(all_loop_region_residues),
                "plddt": aux['plddt'].mean(1)[best_idx],
                "ptm":   aux['ptm'][best_idx],
                "i_pae": aux['losses']['i_pae'][best_idx],
                "i_con": aux['losses']['i_con'][best_idx],
                "i_ptm": aux['i_ptm'][best_idx],
                "loss":  aux['loss'][best_idx]
            }
            
            if args.use_mpnn_loss:
                design_score_dict.update({
                    "mpnn":  aux['losses']['mpnn'][best_idx],
                    "mpnn_seq":  aux['losses']['mpnn_seq'][best_idx],
                })

            result[design_name].update(design_score_dict)
            
            all_chains = ','.join([args.target_chain, binder_chain])
            os.makedirs(os.path.join(output_dir, '03.prediction_outputs'), exist_ok=True)
            
            if args.num_mpnn_designs > 0:                
                mpnn_outputs = mpnn_redesign(
                    x, result, design_name, os.path.join(output_dir, '02.design_outputs', design_name+'.pdb'), os.path.join(output_dir, '03.prediction_outputs'), seed=seed_list[m], 
                    num_designs=args.num_mpnn_designs, model_name=args.mpnn_model_name, weights=args.mpnn_weights
                    )
            
            for i in range(0, args.num_mpnn_designs+1):
                if i > 0:
                    name = '_'.join(design_name.split('_')[:-1]) + '_' + str(i).zfill(3)
                    seq = '/'.join(mpnn_outputs['seq'][i-1].split('/')[-2:])
                    new_seq = '/'.join(output['binder_starting_seq'].split('/')[-2:])
                    result[name] = {'seed': seed_list[m], 'mpnn_redesign_score': mpnn_outputs['score'][i-1], 'redesign_seqid': mpnn_outputs['seqid'][i-1], 'seqid': 1-hamming(list(seq), list(new_seq)), 'seq': seq}
                else:
                    name = design_name

                predict_model, interface_mask = predict(
                    input_args=x, result=result, design_name=name, 
                    template_path=os.path.join(output_dir, '02.design_outputs', design_name+'.pdb'), 
                    input_seq=result[name]['seq'].replace('/', ''), output_dir=os.path.join(output_dir, '03.prediction_outputs'),
                    biasing_mask=loop_biasing_mask, seed=seed_list[m], num_models=2, num_recycles=3
                )

                interface_region = np.where(interface_mask)[0]
                interface_region_residues = [f"{new_pdb['idx']['chain'][x]}{new_pdb['idx']['residue'][x]}" for x in interface_region]
                
                aux = predict_model.aux["all"]
                best_idx = aux['plddt'].mean(1).argmax()
                
                predict_score_dict = {
                    "iface_resid": ','.join(interface_region_residues),
                    "iface_hydropathy": np.mean([kd[x] for x in np.array(list(output['binder_starting_seq'].replace('/','')))[interface_mask]]),
                    "plddt_predict": aux['plddt'].mean(1)[best_idx],
                    "i_ptm_predict": aux['i_ptm'][best_idx],
                    "i_pae_predict": aux['losses']['i_pae'][best_idx],
                    "rmsd_predict":  aux['losses']['rmsd'][best_idx],
                    "ptm_predict":   aux['ptm'][best_idx],
                    "loss_predict":  aux['loss'][best_idx],
                }
                
                result[name].update(predict_score_dict)
                
                os.makedirs(os.path.join(output_dir, '04.scores'), exist_ok=True)
                score_df = pd.DataFrame.from_dict(result, orient="index")
                score_df.to_csv(os.path.join(output_dir, '04.scores', f"{seed}_scores.csv"))

if __name__ == "__main__": 
    parser = create_parser()
    args = parser.parse_args()
    main(args)