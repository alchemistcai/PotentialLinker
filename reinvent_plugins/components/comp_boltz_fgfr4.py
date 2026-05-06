__all__ = ["BoltzScore"]
from collections import defaultdict
import logging
import os
from pathlib import Path
import shutil
import random
import joblib
from pymol import cmd
import numpy as np
import io
from Bio.PDB import  PDBParser, NeighborSearch
from pydantic.dataclasses import dataclass
from util import clean_a3m, smi2inchikey
from reinvent_plugins.components.component_results import ComponentResults
from reinvent_plugins.components.add_tag import add_tag
from reinvent_plugins.normalize import normalize_smiles
from boltz.data.msa.mmseqs2 import run_mmseqs2
from reinvent_plugins.components.boltz_interface import POCKET_CRBN,SEQ_CRBN,POCKET_VHL,SEQ_VHL,run_boltz_lgKd_prediction,generate_boltz_ternary_yaml

logging.basicConfig(level=logging.INFO)
scaler=joblib.load(Path(__file__).parent.parent.parent/'data/scaler.pkl')


def run_superpose_dist_ub_calc(yaml_path: str,generate_bestcif:bool=True,name_e3_origin:str='') -> tuple[str,float,float,int]:
    '''Use P4ward generated CRL complex models to calculate (superposed model filename, min UB-LYS distance, num_clash, closest LYS boltz2 residue index).

    num_clash is defined by: 
    Jofily P, Kalyaanamoorthy S. P4ward: An Automated Modeling Platform for Protac Ternary Complexes. J Chem Inf Model. 2025 Aug 25;65(16):8806-8818. doi: 10.1021/acs.jcim.5c00614. Epub 2025 Aug 13. PMID: 40801829.
    '''
    ypath = Path(yaml_path).absolute()
    min_dist=float('inf')
    num_clash=float('inf')
    closest_lys_resid=-10000
    scoring_items = ('Fail_to_calculate',min_dist,num_clash,closest_lys_resid)
    logging.info(f'Calculating superposing sc items: {ypath.stem}')
    predicted_structure_cif = ypath.parent / f'boltz_results_{ypath.stem}' / 'predictions' / ypath.stem / f'{ypath.stem}_model_0.cif'
    if not os.path.exists(predicted_structure_cif):
        return ('no_3d_cif_file',min_dist,num_clash,closest_lys_resid)
    try:
        cmd.reinitialize()
        cmd.load(predicted_structure_cif, 'ternary')
        lys_poi_sasa_info_after_docking = cmd.get_sasa_relative('chain P and resn LYS',
                                                  subsele='sidechain')
        lys_poi_surf_resis=lys_poi_surf_resis_after_docking = [
            idx_str for (*_, idx_str), sasa_rel in lys_poi_sasa_info_after_docking.items() if sasa_rel >= 0.4
        ]
        # dict[(objname,segi,chain,resi), sasa_rel]
        # relative side-chain SASA >= 40% is on surface (haddock surface cutoff).
        if not lys_poi_surf_resis_after_docking: # fallback to undocked Lys
            cmd.copy_to('poi','ternary and chain P')
            lys_poi_sasa_info_no_docking = cmd.get_sasa_relative('poi and chain P and resn LYS',
                                                    subsele='sidechain')
            lys_poi_surf_resis=lys_poi_surf_resis_no_docking = [
                idx_str for (*_, idx_str), sasa_rel in lys_poi_sasa_info_no_docking.items()
            ]
            if not lys_poi_surf_resis_no_docking:
                logging.debug(str(lys_poi_sasa_info_no_docking))
                return ('no_surf_lys',min_dist,num_clash,closest_lys_resid)
        if name_e3_origin:
            ref_structure_dir=ref_structure_dir = Path(__file__).parent.parent.parent / 'data/crl_models' /name_e3_origin
        elif 'crbn' in str(predicted_structure_cif):
            ref_structure_dir=ref_structure_dir = Path(__file__).parent.parent.parent / 'data/crl_models' /'crbn/'
        elif 'vhl' in str(predicted_structure_cif):
            ref_structure_dir=ref_structure_dir = Path(__file__).parent.parent.parent / 'data/crl_models' /'vhl/'
        if not ref_structure_dir.exists():
            return ('no_UB_E2_cif_file',min_dist,num_clash,closest_lys_resid)

        min_ref_file=''
        for ref_file in os.listdir(ref_structure_dir):
            dist = float('inf')
            if not ref_file.endswith('clean.pdb'):
                continue
            try:
                # distance
                cmd.reinitialize()
                cmd.load(predicted_structure_cif,
                         'ternary')  # reload again to avoid unknown exceptions
                cmd.load(ref_structure_dir / ref_file, 'ref')
                cmd.super('ref and chain C','ternary and chain E') # Don't move ternary!!! Altering Coord will affect SASA calculation
                cmd.select('ubc', 'ref and chain U and resi 75 and name C')
                dist,lys_resi = min(
                    (cmd.get_distance('ubc', f'chain P and resn LYS and name NZ and resi {resi}'),resi)
                    for resi in lys_poi_surf_resis)
                # clash
                cmd.remove('chain C')
                cmd.copy_to('ref', 'ternary')
                parser = PDBParser(QUIET=True)
                st = parser.get_structure('pymol_export', io.StringIO(cmd.get_pdbstr('ref')))
                poi_atoms = st[0]['P'].get_atoms()
                ns_atoms = []
                for chain in st[0]:
                    if chain.id not in [
                            'L', 'E','P'
                    ]:  # these clashes are already calculated in boltz
                        ns_atoms.extend(chain.get_atoms())
                ns = NeighborSearch(list(ns_atoms))
                clash = 0
                for atom in poi_atoms:
                    close_atoms = ns.search(atom.coord, 1)
                    if len(close_atoms) > 0:
                        # then this receptor atom is clashing with at least one atom for the model
                        clash += 1
                if dist<min_dist:
                    min_dist=dist
                    min_ref_file=ref_file
                    num_clash=clash
                    closest_lys_resid=lys_resi
                    if generate_bestcif:
                        cmd.save(ypath.parent/f'{ypath.stem}_best.cif','ref')
            except Exception as e:
                logging.warning(f'Fail to treat {ypath.stem} superposing to {ref_file}')
                logging.warning(e)
                continue
        return (min_ref_file,min_dist,num_clash,closest_lys_resid)
    except Exception as e:
        logging.warning(f'Fail to treat {ypath.stem} calculating superposing scores.')
        logging.warning(e)
        return scoring_items

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
    name_e3_origin:str=''
)-> tuple[float,dict[str,float|str]]:
    '''Calculate Harmonic(lgKD_ternary, dist_ub_square,num_clash), each one is normalized. 
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
    super_scores=run_superpose_dist_ub_calc(yaml_ternary,generate_bestcif,name_e3_origin)
    dist_ub_square=super_scores[1]**2
    num_clash=super_scores[2]
    if dist_ub_square>10000:
        logging.warning(f'dist_ub is bigger than 100, the reference structure / fail reason is {super_scores[0]}')
    if np.any(np.isinf([[lgKd_ternary,dist_ub_square,num_clash]])):
        score=0
    else:
        score=float(3/((1/(1-scaler.transform([[lgKd_ternary,dist_ub_square,num_clash]])+1e-10)).sum(axis=1)))
    logging.info(f'lgKd_ternary: {lgKd_ternary:.3f},dist_ub_square: {dist_ub_square:.3f}, num_clash: {num_clash}, closest LYS resid: {super_scores[3]}, harmonic score: {score:.3f}')
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