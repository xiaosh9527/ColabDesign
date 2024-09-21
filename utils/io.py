import numpy as np
import jax
import jax.numpy as jnp

from colabdesign.shared.utils import copy_dict
from colabdesign.af.prep import prep_pdb
from colabdesign.af.alphafold.common import protein, residue_constants
from colabdesign.shared.protein import renum_pdb_str

PDB_CHAIN_IDS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'

def rank_array(input_array):
    # numpy.argsort returns the indices that would sort an array.
    # We convert it to a python list before returning
    return list(np.argsort(input_array))
    
def rank_and_write_pdb(af_model, name, write_all=False, renum_pdb=True):
    ranking = rank_array(af_model.aux["all"]["loss"])
    if write_all != True:
        ranking = [ranking[0]]

    aux = af_model._tmp["best"]["aux"]
    aux = aux["all"]

    p = {k: aux[k] for k in ["aatype", "residue_index", "atom_positions", "atom_mask"]}
    unique_chain_ids = np.unique(af_model._pdb["idx"]["chain"])
    chain_id_mapping = {cid: n for n, cid in enumerate(list(PDB_CHAIN_IDS))}
    chain_index = np.array([chain_id_mapping[cid] for cid in af_model._pdb["idx"]["chain"]])
    p["chain_index"] = np.r_[[chain_index], [chain_index]]
    p["b_factors"] = 100 * p["atom_mask"] * aux["plddt"][..., None]

    def to_pdb_str(x, n=None):
        p_str = protein.to_pdb(protein.Protein(**x))
        p_str = "\n".join(p_str.splitlines()[1:-2])
        if renum_pdb:
            p_str = renum_pdb_str(p_str, af_model._lengths)
        if n is not None:
            p_str = f"MODEL{n:8}\n{p_str}\nENDMDL\n"
        return p_str

    for n in ranking:
        p_str = ""
        p_str += to_pdb_str(jax.tree_map(lambda x: x[n], p), None)
        p_str += "END\n"

        with open(name, "w") as f:
            f.write(p_str)

def rank_array_predict(input_array):
    # numpy.argsort returns the indices that would sort an array.
    # We convert it to a python list before returning
    return list(np.argsort(input_array))[::-1]
    
def rank_and_write_pdb_predict(af_model, name, write_all=False, renum_pdb=True):
    ranking = rank_array_predict(np.mean(af_model.aux["all"]["plddt"], -1))
    if write_all != True:
        ranking = [ranking[0]]
    
    aux = af_model.aux
    aux = aux["all"]

    p = {k: aux[k] for k in ["aatype", "residue_index", "atom_positions", "atom_mask"]}
    unique_chain_ids = np.unique(af_model._pdb["idx"]["chain"])
    chain_id_mapping = {cid: n for n, cid in enumerate(list(PDB_CHAIN_IDS))}
    chain_index = np.array([chain_id_mapping[cid] for cid in af_model._pdb["idx"]["chain"]])
    p["chain_index"] = np.r_[[chain_index], [chain_index]]
    p["b_factors"] = 100 * p["atom_mask"] * aux["plddt"][..., None]

    def to_pdb_str(x, n=None):
        p_str = protein.to_pdb(protein.Protein(**x))
        p_str = "\n".join(p_str.splitlines()[1:-2])
        if renum_pdb:
            p_str = renum_pdb_str(p_str, af_model._lengths)
        if n is not None:
           p_str = f"MODEL{n:8}\n{p_str}\nENDMDL\n"
        return p_str

    for n in ranking:
        p_str = ""
        p_str += to_pdb_str(jax.tree_map(lambda x: x[n], p), None)
        p_str += "END\n"

        with open(name, "w") as f:
            f.write(p_str)

