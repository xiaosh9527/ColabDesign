import os, re, sys, string, argparse, jax, subprocess, tempfile, Bio, shutil, mrcfile
import numpy as np
import pandas as pd
import jax.numpy as jnp

from scipy.special import softmax
from scipy.spatial import cKDTree 
from scipy.spatial.distance import cdist, hamming

from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.mpnn import mk_mpnn_model
from colabdesign.shared.prep import prep_pos
from colabdesign.shared.utils import copy_dict
from colabdesign.af.prep import prep_pdb
from colabdesign.af.alphafold.common import protein, residue_constants
from colabdesign.shared.protein import renum_pdb_str

from Bio.PDB import *
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.Selection import unfold_entities
from Bio import SeqUtils
from Bio.Align import substitution_matrices

from abnumber import Chain

import warnings

def process_ab(
    input_path: str, 
    output_path: str,
    chains: str,
    mpnn_fixed_regions = None,
    mpnn_weights: str = 'abmpnn',
    mpnn_model_name: str = 'abmpnn',
    seed: int = 42
):
    AA3to1_covert = SeqUtils.IUPACData.protein_letters_3to1
    AA3to1_covert = {k.upper(): v for k,v in AA3to1_covert.items()}

    parser = PDBParser(QUIET=True)
    ab_struct = parser.get_structure("ab", input_path)

    mpnn_model = mk_mpnn_model(model_name=mpnn_model_name, weights=mpnn_weights, seed=seed)
    mpnn_model.prep_inputs(pdb_filename=input_path, chain=chains, fix_pos=mpnn_fixed_regions, inverse=True)
    O = mpnn_model.sample(num=1, batch=1, temperature=0.1, rescore=False)
    seqs = O['seq'][0]
    
    ab_seq = []
    for chain, seq in zip(chains.split(','), seqs.split('/')):        
        abnumber_chain = Chain(seq, scheme='chothia')
        fab = re.search(abnumber_chain.seq, seq)
        mask = np.ones(len(seq), dtype=bool)
        mask[np.arange(*fab.span())] = 0.0
        
        for c in ab_struct[0]:
            if c.get_id() == chain:
                all_res_id = [r.get_id()[1] for r in c.get_residues()]
                for i in np.where(mask)[0]:
                    c.detach_child((' ', all_res_id[i], ' '))
        
        ab_seq.append(abnumber_chain.seq)
    
    io = PDBIO()
    io.set_structure(ab_struct)
    io.save(output_path)
    
    return '/'.join(ab_seq)
    
def combine_pdbs(
    target_pdb: str, target_chain: str, 
    binder_pdb: str, binder_chain: str, 
    binder_pdb_processed: str,
    output_pdb_path: str, 
    seed: int = 42,
    mpnn_fixed_regions = None,
    mpnn_weights: str = 'abmpnn',
    mpnn_model_name: str = 'abmpnn',
    ):
    
    ''' Combine the target and binder PDB files into a single PDB file '''
    
    output = {}
    new_chain_name_list = list(string.ascii_uppercase)

    ab_seq = process_ab(binder_pdb, binder_pdb_processed, binder_chain, mpnn_fixed_regions=mpnn_fixed_regions, mpnn_weights=mpnn_weights, mpnn_model_name=mpnn_model_name, seed=seed)
    
    # Read the PDB file
    parser = PDBParser(QUIET=True)
    target_struct = parser.get_structure("target", target_pdb)
    binder_struct = parser.get_structure("binder", binder_pdb_processed)
    target_chains = [chain for chain in target_struct.get_chains() if chain.id in target_chain.split(',')]
    binder_chains = [chain for chain in binder_struct.get_chains() if chain.id in binder_chain.split(',')]
    target_chains_id = [chain.id for chain in target_chains]
    binder_chains_id = [chain.id for chain in binder_chains]
    
    # NOTE: Renumber the binder chain ID to avoid chain name reoccurancebiasing_mask
    if np.in1d(target_chains_id, binder_chains_id).any():
        binder_chains_id = []
        for chain in target_chains_id:
            new_chain_name_list.remove(chain)
        for idx, chain in enumerate(binder_chains): 
            new_chain_id = new_chain_name_list[idx]
            chain.id = new_chain_id
            binder_chains_id.append(new_chain_id)
            
    sb = StructureBuilder()
    sb.init_structure("combined")
    sb.init_model(0)
    combined_struct = sb.get_structure()
    for idx, chain in enumerate(target_chains+binder_chains):
        combined_struct[0].add(chain)

    '''Combining the chains into a single PDB file'''    
    class NonHetSelect(Select):        
        '''Selects only non-hetero atoms'''
        def accept_residue(self, residue):            
            return 1 if residue.id[0] == " " else 0

    io = PDBIO()
    io.set_structure(combined_struct)
    io.save(output_pdb_path)
    io.save(output_pdb_path, NonHetSelect())

    output.update(
        {
            "target_chain": ','.join(target_chains_id),
            "binder_chain": ','.join(binder_chains_id), 
            "binder_starting_seq": ab_seq,
        }
    )
    
    return output

def add_mpnn_loss(self, model_name='abmpnn', weights='abmpnn', mpnn=0.2, mpnn_seq=0.1, seed=42):
    '''
    add mpnn loss
    weights: ['original', 'soluble', 'abmpnn']
    mpnn = maximize confidence of proteinmpnn
    mpnn_seq = push designed sequence to match proteinmpnn logits
    '''
    self._mpnn = mk_mpnn_model(model_name=model_name, weights=weights, seed=seed)
    def loss_fn(inputs, outputs, aux, key):
        # get structure
        atom_idx = tuple(residue_constants.atom_order[k] for k in ["N","CA","C","O"])
        I = {"S":           inputs["aatype"],
                "residue_idx": inputs["residue_index"],
                "chain_idx":   inputs["asym_id"],
                "X":           outputs["structure_module"]["final_atom_positions"][:,atom_idx],
                "mask":        outputs["structure_module"]["final_atom_mask"][:,1],
                "lengths":     self._lengths,
                "key":         key}

        if "offset" in inputs:
            I["offset"] = inputs["offset"]

        # set autoregressive mask
        L = sum(self._lengths)
        if self.protocol == "binder":
            I["ar_mask"] = 1 - np.eye(L)
            I["ar_mask"][-self._len:,-self._len:] = 0
        else:
            I["ar_mask"] = np.zeros((L,L))

        # get logits
        logits = self._mpnn._score(**I)["logits"][:,:20]
        if self.protocol == "binder":
            logits = logits[-self._len:]
        else:
            logits = logits[:self._len]
        aux["mpnn_logits"] = logits

        # compute loss
        log_q = jax.nn.log_softmax(logits)
        p = inputs["seq"]["hard"]
        q = jax.nn.softmax(logits)
        losses = {}
        losses["mpnn"] = -log_q.max(-1).mean()
        losses["mpnn_seq"] = -(p * jax.lax.stop_gradient(log_q)).sum(-1).mean()
        return losses

    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["mpnn"] = mpnn
    self.opt["weights"]["mpnn_seq"] = mpnn_seq

def design(
    input_args: dict,
    design_name: str,
    output_dir: str,
    binder_starting_seq: str, # NOTE: starting sequence (used for calculating seqid) can be MPNN designed seq
    rm_aa: str,               # NOTE: to prevent given AA types in all positions
    biasing_aa: str,          # NOTE: to bias against given AA types in the selected positions
    biasing_mask,             # NOTE: selected positions for biasing against
    pssm_steps: int = 120,
    semi_greedy_steps: int = 24,
    seed: int = 0, 
    lr: float = 2e-2, 
    num_models: int = 2, 
    num_recycles: int = 1
    ):
    '''
    Design binder sequence using PSSM-semigreedy optimization
    '''    
    clear_mem()
    af_model = mk_afdesign_model(
        protocol="binder",
        use_multimer=input_args["use_multimer"],
        num_recycles=num_recycles,
        data_dir="/work/lpdi/users/shxiao/params/",
    )
    af_model.prep_inputs(**input_args, ignore_missing=False)
    
    add_dist_loss(af_model, input_args['binder_chain'], input_args['target_chain'], input_args['map_path'], 0.1, input_args['density_cutoff'])

    if args.use_mpnn_loss: add_mpnn_loss(af_model, weights=args.mpnn_weights, model_name=args.mpnn_model_name, mpnn=0.1, mpnn_seq=0.0, seed=seed)

    # NOTE: Important step: fix binder positions
    af_model._pdb["len"] = sum(af_model._pdb["lengths"])
    af_model.opt["pos"] = af_model._pdb["pos"] = np.arange(af_model._pdb["len"])
    af_model._pos_info = {"length":np.array([af_model._pdb["len"]]), "pos":af_model._pdb["pos"]}

    sub_fix_pos = []
    sub_i = []
    pos = af_model.opt["pos"].tolist()
    residues = af_model._pdb["idx"]['residue'][np.in1d(af_model._pdb["idx"]["chain"], input_args['binder_chain'].split(','))]
    chains = af_model._pdb["idx"]['chain'][np.in1d(af_model._pdb["idx"]["chain"], input_args['binder_chain'].split(','))]

    if args.fix_pos:
        # NOTE: IMPORTANT for fixing N-term seq of Mob!!!
        for i in prep_pos(args.fix_pos, residues, chains)["pos"]:
            if i in pos:
                sub_i.append(i)
                sub_fix_pos.append(pos.index(i))
        sub_i += af_model._target_len
        af_model.opt["fix_pos"] = np.array(sub_fix_pos)
        af_model._wt_aatype_sub = af_model._pdb["batch"]["aatype"][sub_i]
    
    af_model.opt["weights"].update({"dgram_cce":0.1, "plddt":0.0, "rmsd":0.2, "con":0.0, "i_con":1.0, "i_pae":0.0})
    print("weights", af_model.opt["weights"])
    print(f'\nNow designing {design_name}...\n[Random seed {seed}]: Designing binder sequence using PSSM-semigreedy optimization...')
    af_model.restart(seq=binder_starting_seq, rm_aa=rm_aa, seed=seed, reset_opt=False)
    af_model.set_optimizer(optimizer="adam", learning_rate=lr, norm_seq_grad=True)

    # NOTE: Prevent templating the loop regions !!!
    (T,L,rm) = (af_model._lengths[0],sum(af_model._lengths),{})
    rm_opt_list = ['rm_template', 'rm_template_seq', 'rm_template_sc']
    for n in rm_opt_list:
        rm[n] = np.full(L,False)
        rm[n][-af_model._binder_len:] = biasing_mask

    af_model._inputs.update(rm)
    
    models = af_model._model_names[:num_models]
    flags = {"num_recycles":num_recycles,
            "models":models,
            "dropout":True}
    
    af_model.design_pssm_semigreedy(pssm_steps, semi_greedy_steps, **flags)
    rank_and_write_pdb(af_model, name=os.path.join(output_dir, f"{design_name}.pdb"), write_all=False, renum_pdb=False)

    return af_model

def mpnn_redesign(
    input_args: dict,
    result: dict,
    design_name: str,
    template_path: str, 
    output_dir: str,
    seed: int = 0, 
    num_designs: int = 8,
    sampling_temp: float = 0.2,
    rm_aa: str = 'C',               # NOTE: to prevent given AA types in all positions
    model_name: str = 'abmpnn',
    weights: str = 'abmpnn',
):      
    clear_mem()
    mpnn_model = mk_mpnn_model(model_name=model_name, weights=weights, seed=seed)        
    
    mpnn_model.prep_inputs(
        pdb_filename=template_path,
        chain=','.join([input_args['target_chain'], input_args['binder_chain']]),
        fix_pos=input_args['target_chain'],
        inverse=False,
        rm_aa=rm_aa,
        verbose=False
    )

    def find_interface(mpnn_model, binder_chain: str):
        binder_chain_id = np.in1d(mpnn_model.pdb['idx']['chain'], binder_chain.split(','))
        binder_atom_coords = mpnn_model.pdb['cb_feat']['atoms'][binder_chain_id]
        target_atom_coords = mpnn_model.pdb['cb_feat']['atoms'][~binder_chain_id]
        
        target_ckdtree = cKDTree(target_atom_coords) 
        d,r = target_ckdtree.query(binder_atom_coords)
        
        interface_masking = (d<=7.0)
    
        return interface_masking

    interface_mask = find_interface(mpnn_model, input_args['binder_chain'])
    # interface_mask[:24] = 0.0  # NOTE: Make sure the first 24 is not selected
    binder_chain_id = np.in1d(mpnn_model.pdb['idx']['chain'], input_args['binder_chain'].split(','))

    binder_bias = mpnn_model._inputs["bias"][np.where(binder_chain_id)]
    for i in np.where(~interface_mask)[0]:
        ori_aa = mpnn_model.pdb['batch']['aatype'][binder_chain_id][i]    
        binder_bias[i, ori_aa] = 1e7
    
    mpnn_model._inputs["bias"][np.where(binder_chain_id)] = binder_bias
    
    out = mpnn_model.sample(num=num_designs, temperature=sampling_temp)

    return out

def predict(
    input_args: dict,
    result: dict,
    design_name: str,
    template_path: str,
    input_seq: str, 
    output_dir: str,
    seed: int = 0, 
    num_models: int = 2, 
    num_recycles: int = 3,
    **kwargs
    ):
    '''
    Predicting with AF2Rank with loop being masked
    '''
    clear_mem()
    af_model = mk_afdesign_model(
        protocol="binder",
        use_multimer=True,
        num_recycles=args.num_recycles,
        data_dir="/work/lpdi/users/shxiao/params/",    
    )
    
    print(f'Now predicting {design_name} with AF2Rank...')    
    
    af_model.prep_inputs(
        pdb_filename=template_path, 
        target_chain=input_args['target_chain'],
        binder_chain=input_args['binder_chain'],
        initial_guess=True,
        rm_target=False,
        rm_target_seq=False,
        rm_target_sc=False,
        rm_binder=True,
        rm_binder_seq=True,
        rm_binder_sc=True,
        rm_template_ic=True
    )

    def find_interface(af_model, binder_chain: str):
        binder_chain_id = np.in1d(af_model._pdb['idx']['chain'], binder_chain.split(','))
        binder_atom_coords = af_model._pdb['cb_feat']['atoms'][binder_chain_id]
        target_atom_coords = af_model._pdb['cb_feat']['atoms'][~binder_chain_id]
        
        target_ckdtree = cKDTree(target_atom_coords) 
        d,r = target_ckdtree.query(binder_atom_coords)
        
        interface_masking = (d<=7.0)
    
        return interface_masking

    interface_mask = find_interface(af_model, input_args['binder_chain'])
    
    if 'biasing_mask' in kwargs:
        biasing_mask = kwargs['biasing_mask']
        all_mask = biasing_mask | interface_mask

    print(f'Masking {len(np.where(all_mask)[0])} residues on binder for prediction...')
    af_model._inputs['batch']['all_atom_mask'][-af_model._binder_len:][all_mask, :] = np.zeros_like(af_model._inputs['batch']['all_atom_mask'][-af_model._binder_len:][all_mask, :])
    
    models = af_model._model_names[:num_models]
    
    flags = {
        "num_recycles":num_recycles,
        "models":models,
        "dropout":True
    }

    af_model.predict(seq=input_seq, return_aux=False, verbose=True, seed=seed, **flags)
    rank_and_write_pdb_predict(af_model, name=os.path.join(output_dir, f"{design_name}.pdb"), write_all=False, renum_pdb=False)

    print(f'\nDone.\n')    
    
    return af_model, interface_mask

def partition_empem_density(self, binder_chain, target_chain, map_path: str, density_cutoff: float = 0.2):
    '''
    This function takes the antibody template and the electrion density map and returns the density points 
    that are closest to the antibody backbone
    '''
    atom_coords = self._pdb['batch']['all_atom_positions']
    binder_chain_id = np.in1d(self._pdb['idx']['chain'], binder_chain.split(','))
    target_chain_id = np.in1d(self._pdb['idx']['chain'], target_chain.split(','))

    target_atom_coords = atom_coords[target_chain_id][np.where(self._pdb['batch']['all_atom_mask'][target_chain_id])].reshape(-1, 3)

    xyz, emd = read_mrc(map_path)
    density_sele_idx = np.where(emd>=density_cutoff)
    emd_values = emd[np.where(emd>=density_cutoff)]
    dim = int(np.cbrt(xyz.shape[0]))
    map_coords = xyz.reshape(dim, dim, dim, 3)[np.where(emd>=density_cutoff)]
        
    map_cdktree = cKDTree(map_coords)
    d1, r1 = map_cdktree.query(target_atom_coords, k=100)
    
    target_sele_idx = np.unique(r1[np.where(d1<=3.0)])
    target_sele = np.zeros(len(map_coords), dtype=bool)
    target_sele[target_sele_idx] = 1.0
    binder_sele = ~target_sele
    
    return map_coords[binder_sele], map_coords[target_sele]

def read_mrc(mrcfilename):
    """
    Read a mrc file and return the xyz and density values at the given level
    if given
    """
    xyz = []
    with mrcfile.open(mrcfilename) as emd:
        nx, ny, nz = emd.header['nx'], emd.header['ny'], emd.header['nz']
        x, y, z = emd.header['origin']['x'], emd.header['origin']['y'], emd.header['origin']['z']
        dx, dy, dz = emd.voxel_size['x'], emd.voxel_size['y'], emd.voxel_size['z']
        xyz = np.meshgrid(np.arange(x, x+nx*dx, dx),
                             np.arange(y, y+ny*dy, dy),
                             np.arange(z, z+nz*dz, dz),
                             indexing='ij')
        xyz = np.asarray(xyz)
        xyz = xyz.reshape(3, nx*ny*nz)
        xyz = xyz.T
        
        return xyz, emd.data.flatten(order='F').reshape(nx, ny, nz)

def add_dist_loss(self, binder_chain, target_chain, map_path, weight: float = 1.0, density_cutoff: float = 0.2):
    binder_em_coords, target_em_coords = partition_empem_density(self, binder_chain, target_chain, map_path, density_cutoff)
    def loss_fn(inputs, outputs, aux, key):
        atom_coords = outputs["structure_module"]["final_atom_positions"]
        binder_chain_id = np.in1d(self._pdb['idx']['chain'], binder_chain.split(','))
        binder_coords = atom_coords[binder_chain_id][:, residue_constants.atom_order["CA"]].reshape(-1,3)
        d_m = jnp.sqrt(jnp.sum((binder_coords[:, None, :] - binder_em_coords[None, :, :])**2, axis=-1))

        return {"dist": d_m.min(-1).mean()}
    
    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["dist"] = weight

# from cryoem_maps_utils import *

# def add_dist_loss(self, binder_chain, target_chain, em_map, weight: float = 1.0, density_cutoff: float = 0.2):     
    
#     def loss_fn(inputs, outputs, aux, key):
#         atom_coords = outputs["structure_module"]["final_atom_positions"]
        
#         binder_chain_id = np.in1d(self._pdb['idx']['chain'], binder_chain.split(','))
#         binder_coords = atom_coords[binder_chain_id][:, residue_constants.atom_order["CA"]].reshape(-1,3)

#         loss = 0
#         for coord in binder_coords:
#             loss += em_map.get_density_coordinate(*coord)
        
#         return {"dist": loss}
    
#     self._callbacks["model"]["loss"].append(loss_fn)
#     self.opt["weights"]["dist"] = weight
