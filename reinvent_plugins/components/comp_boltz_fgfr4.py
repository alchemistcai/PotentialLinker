__all__ = ["BoltzScore"]
from collections import defaultdict
from functools import partial
import logging
import os
from pathlib import Path
import shutil
from typing import Callable
import joblib
import numpy as np
from pydantic.dataclasses import dataclass
from util import clean_a3m, smi2inchikey
from reinvent_plugins.components.component_results import ComponentResults
from reinvent_plugins.components.add_tag import add_tag
from reinvent_plugins.normalize import normalize_smiles
from boltz.data.msa.mmseqs2 import run_mmseqs2
from reinvent_plugins.components.boltz_interface import POCKET_CRBN,SEQ_CRBN,POCKET_VHL,SEQ_VHL, IntOrInf,run_boltz_lgKd_prediction,generate_boltz_ternary_yaml, run_superpose_scoring_items_calc

logging.basicConfig(level=logging.INFO)
scaler=joblib.load(Path(__file__).parent.parent.parent/'data/scaler.pkl')

run_superpose_calc_best_nobsa:Callable[...,tuple[str,float,IntOrInf,int]]=partial(run_superpose_scoring_items_calc,best_record_only=True,calc_bsa=False)
'''Align a given yaml file's prediction structure to P4ward CRL reference, and return [uperposed model filename, min UB-LYS distance, clash atoms,closest lys resid].

If name_e3_origin is given, use your E2/Ub/E3 reference directory instead.
'''

def calculate_one_affinity_score(
    lig_smi: str,
    name_poi: str,
    seq_poi: str,
    pocket_poi: list[int],
    name_e3: str,
    seq_e3: str|None=None,
    pocket_e3: list[int]|None=None,
    output_path: str='data/precompute',
    msa_path: str = 'data/MSA',
    contact_ppi: list[tuple[int, int]] | None = None,
    template_cif_or_path:str='',
    template_chain_poi_can:str='',
    template_chain_e3_can:str='',
    extract_from_precomputed:bool=False,
    is_covalent_poi:bool=False,
    lig_poi_part:str='N(c1c(c(cc(c1))C)Nc1nc2c(cn1)cc(cc2)c1c(Cl)c(OC)cc(OC)c1Cl)C(=O)CC',
    lig_warhead_smarts: str='[C:1]-C-C=O', 
    covalent_resid_can: int= 104, 
    covalent_res_atomname: str='SG',
    generate_bestcif:bool=True,
    scaler=scaler,
    name_e3_origin:str='',
    w_lgkd:float=1,
    w_d2:float=1,
    w_clash:float=1
)-> tuple[float,dict[str,float|str]]:
    '''Calculate NWHM(lgKD_ternary, dist_ub_square,num_clash), each one is normalized. 
    '''
    inchikey=smi2inchikey(lig_smi)
    if not seq_e3:
        if name_e3=='vhl':
            seq_e3=SEQ_VHL
        elif name_e3=='crbn':
            seq_e3=SEQ_CRBN
        else:
            logging.warning(f'Sequence of the given e3 {name_e3} is not given and e3 is not crbn or vhl. May lead to failure.')
    if not pocket_e3:
        if name_e3=='vhl':
            pocket_e3=POCKET_VHL
        elif name_e3=='crbn':
            pocket_e3=POCKET_CRBN
        else:
            logging.warning(f'Pocket of the given e3 {name_e3} is not given and e3 is not crbn or vhl. May lead to failure.')
    if not os.path.exists(output_path):
        os.makedirs(output_path,exist_ok=True)
    msa_poi = os.path.join(msa_path, name_poi + '.a3m')
    msa_e3 = os.path.join(msa_path, name_e3 + '.a3m')
    yaml_ternary = os.path.join(
        output_path, f'{name_poi}_{name_e3}_{inchikey}_ternary.yaml')
    generate_boltz_ternary_yaml(lig_smi, seq_poi, pocket_poi, msa_poi, seq_e3,
                                pocket_e3, msa_e3, yaml_ternary,is_covalent_poi,lig_poi_part,lig_warhead_smarts,covalent_resid_can,covalent_res_atomname,contact_ppi,template_cif_or_path,template_chain_poi_can,template_chain_e3_can)
    lgKd_ternary=run_boltz_lgKd_prediction(yaml_ternary,extract_from_precomputed)
    super_scores=run_superpose_calc_best_nobsa(yaml_ternary,generate_bestcif=generate_bestcif,name_e3_origin=name_e3_origin)
    dist_ub_square=super_scores[1]**2
    num_clash=super_scores[2]
    if dist_ub_square>10000:
        logging.warning(f'dist_ub is bigger than 100, the reference structure / fail reason is {super_scores[0]}')
    if np.any(np.isinf([[lgKd_ternary,dist_ub_square,num_clash]])):
        score=0
    else:
        ws=np.array((w_lgkd,w_d2,w_clash))
        score=float(ws.sum()/((ws/(1-scaler.transform([[lgKd_ternary,dist_ub_square,num_clash]])+1e-10)).sum()))
    logging.info(f'lgKd_ternary: {lgKd_ternary:.3f},dist_ub_square: {dist_ub_square:.3f}, num_clash: {num_clash}, closest LYS resid: {super_scores[3]}, NWHM score: {score:.3f}')
    return score,{'lgKd_ternary':lgKd_ternary,'dist_ub_square':dist_ub_square,'dist_ub':super_scores[1],'num_clash':num_clash,'ref_model':super_scores[0],'closest_lys_resid':super_scores[3]}

logger = logging.getLogger(__name__)

@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the scoring component

    Note that all parameters are always lists because components can have
    multiple endpoints and so all the parameters from each endpoint is
    collected into a list.  This is also true in cases where there is only one
    endpoint.
    """

    poi_name: list[str]
    poi_seq: list[str]
    poi_pocket: list[list[int]]
    e3_name: list[str]
    calculate_path: list[str]
    is_covalent_poi: list[bool]
    lig_poi_part : list[str]|None=None
    lig_warhead_smarts: list[str]|None=None
    covalent_resid_can: list[int]|None=None
    covalent_res_atomname: list[str]|None=None
    generate_bestcif:list[bool]|None=None
    w_lgkd:list[float]|None=None
    w_d2:list[float]|None=None
    w_clash:list[float]|None=None



@add_tag("__component")
class BoltzScore:
    """Adjust according to DockStream (https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00563-7) and Boltz-2.

    Consistent with previous behaviour, cases where no docking pose is produced
    are given a score of zero
    """

    def __init__(self,parameter:Parameters):
        self._internal_step = 0
        self.poi_name=parameter.poi_name[0]
        self.poi_seq=parameter.poi_seq[0]
        self.poi_pocket=parameter.poi_pocket[0]
        self.e3_name=parameter.e3_name[0]
        self.calculate_path=parameter.calculate_path[0] # dir saving poi_name.msa, boltz cif and json files
        self.is_covalent_poi=is_covalent=parameter.is_covalent_poi[0]
        self.lig_poi_part =parameter.lig_poi_part[0] if is_covalent else None
        self.lig_warhead_smarts=parameter.lig_warhead_smarts[0] if is_covalent else None
        self.covalent_resid_can=parameter.covalent_resid_can[0] if is_covalent else None
        self.covalent_res_atomname=parameter.covalent_res_atomname[0] if is_covalent else None
        self.generate_bestcif=True if parameter.generate_bestcif is None else parameter.generate_bestcif[0]
        self.w_lgkd=1 if parameter.w_lgkd is None else parameter.w_lgkd[0] 
        self.w_d2=1 if parameter.w_d2 is None else parameter.w_d2[0] 
        self.w_clash=1 if parameter.w_clash is None else parameter.w_clash[0] 
        self.smiles_type = "rdkit_smiles"
        calculate_prefix=os.path.join(self.calculate_path,self.poi_name)
        os.makedirs(self.calculate_path,exist_ok=True)
        if not os.path.exists(os.path.join(calculate_prefix,'crbn.a3m')):
            shutil.copy(Path(__file__).parent.parent.parent/'data/MSA/crbn.a3m',self.calculate_path+'/crbn.a3m')
            shutil.copy(Path(__file__).parent.parent.parent/'data/MSA/vhl.a3m',self.calculate_path+'/vhl.a3m')
        
        if not os.path.exists(calculate_prefix+'_env/'):
            run_mmseqs2(self.poi_seq,prefix=calculate_prefix,)
            tmp_filepath=calculate_prefix+'_tmp.a3m' 
            msa_path=calculate_prefix+'.a3m'
            shutil.move(calculate_prefix+'_env/bfd.mgnify30.metaeuk30.smag30.a3m',tmp_filepath)
            clean_a3m(tmp_filepath,msa_path)

    @normalize_smiles
    def __call__(self, smilies: list[str]) -> np.array:
        results = [calculate_one_affinity_score(smi,self.poi_name,self.poi_seq,self.poi_pocket,self.e3_name,output_path=self.calculate_path,msa_path=self.calculate_path,is_covalent_poi=self.is_covalent_poi,lig_poi_part=self.lig_poi_part,lig_warhead_smarts=self.lig_warhead_smarts,covalent_resid_can=self.covalent_resid_can,covalent_res_atomname=self.covalent_res_atomname,generate_bestcif=self.generate_bestcif) for smi in smilies]
        scores = [result[0] for result in results]
        metadata = defaultdict(list)
        for result in results:
            for key,item in result[1].items():
                metadata[key].append(item)
        self._internal_step += 1
        return ComponentResults([scores],metadata=metadata) # FIXME: recalculate it because of a new field