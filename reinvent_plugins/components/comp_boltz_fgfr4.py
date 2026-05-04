__all__ = ["BoltzScore"]
from collections import defaultdict
import logging
from math import isinf
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
from rdkit.Chem import MolToInchiKey,MolFromSmiles,AddHs,MolFromSmarts,AssignStereochemistry,AllChem
from util import clean_a3m
from reinvent_plugins.components.component_results import ComponentResults
from reinvent_plugins.components.add_tag import add_tag
from reinvent_plugins.normalize import normalize_smiles
from boltz.data.msa.mmseqs2 import run_mmseqs2
from reinvent_plugins.components.boltz_interface import POCKET_CRBN,SEQ_CRBN,POCKET_VHL,SEQ_VHL,run_boltz_lgKd_prediction

logging.basicConfig(level=logging.INFO)
scaler=joblib.load(Path(__file__).parent.parent.parent/'data/scaler.pkl')
Ws=(-0.4427,-0.05851,-0.2191,0.3496)
def generate_boltz_covalent_or_not_ternary_yaml(
    lig_smi: str,
    seq_poi: str,
    pocket_poi: list[int],
    msa_poi: str,
    seq_e3: str,
    pocket_e3: list[int],
    msa_e3: str,
    output_filename: str,
    is_covalent_poi: bool,
    lig_poi_part:str='N(c1c(c(cc(c1))C)Nc1nc2c(cn1)cc(cc2)c1c(Cl)c(OC)cc(OC)c1Cl)C(=O)CC', # FGFR4 BLU9931
    lig_warhead_smarts: str='[C:1]-C-C=O', #  reaction site after addition
    covalent_resid_can: int= 104, # (CYS 552) - offset 448
    covalent_res_atomname: str='SG',
):
    pocket_poi_input = ','.join(f'[P, {resid}]' for resid in pocket_poi)
    pocket_e3_input = ','.join(f'[E, {resid}]' for resid in pocket_e3)
    covalent_constraint=''
    if is_covalent_poi:
        tmp_mol=MolFromSmiles(lig_smi)
        if tmp_mol is None:
            logging.error(f'Given lig_smi {lig_smi} is not a valid smiles.')
            return None
        tmp_mol = AddHs(tmp_mol)
        poi_mol=MolFromSmiles(lig_poi_part)
        if poi_mol is None:
            logging.error(f'Given lig_smi {lig_smi} fails to add Hs.')
            return None
        poi_match=tmp_mol.GetSubstructMatch(poi_mol)
        if not poi_match:
            logging.error(f'Given Poi part substructure {lig_poi_part} not found.')
            return None
        poi_atom_ids=set(poi_match)
        pattern=MolFromSmarts(lig_warhead_smarts)
        matches=tmp_mol.GetSubstructMatches(pattern)
        if not matches:
            logging.error(f'Given Poi covalent warhead SMARTS {lig_warhead_smarts} not found.')
            return None
        poi_matches=[]
        for match in matches:
            if match[0] in poi_atom_ids:
                all_in_poi = all(idx in poi_atom_ids for idx in match)
                if all_in_poi:
                    poi_matches.append(match)
        if not poi_matches:
            logging.error(f'Given Poi covalent warhead SMARTS {lig_warhead_smarts} not in Poi part substructure {lig_poi_part}.')
            return None
        selected_match = random.choice(poi_matches)
        target_atom_idx = selected_match[0]
        canonical_order = AllChem.CanonicalRankAtoms(tmp_mol) # get atom name according to boltz schema
        AssignStereochemistry(tmp_mol, force=True, cleanIt=True)
        atom_name_to_idx = {}
        for atom, can_idx in zip(tmp_mol.GetAtoms(), canonical_order):
            atom_name = atom.GetSymbol().upper() + str(can_idx + 1)
            atom_name_to_idx[atom.GetIdx()] = atom_name
        target_atom_name = atom_name_to_idx.get(target_atom_idx)
        covalent_constraint=f"""\n  - bond:
      atom1: [L, 1, {target_atom_name}]
      atom2: [P, {covalent_resid_can}, {covalent_res_atomname}]"""

    template = f'''sequences:
  - protein:
      id: P
      sequence: {seq_poi}
      msa: {os.path.abspath(msa_poi) if seq_poi!=seq_e3 else os.path.abspath(msa_e3)}
  - protein:
      id: E
      sequence: {seq_e3}
      msa: {os.path.abspath(msa_e3)}
  - ligand:
      id: L
      smiles: '{lig_smi}'
constraints:{covalent_constraint}
  - pocket:
      binder: L
      contacts: [ {pocket_poi_input} ,{pocket_e3_input} ]
      max_distance: 4
      force: true
properties:
  - affinity:
      binder: L
'''
    with open(output_filename, 'w') as f:
        f.write(template)

def run_superpose_dist_ub_calc(yaml_path: str,generate_bestcif:bool=True) -> tuple[str,float,float]:
    '''Use P4ward generated CRL complex models to calculate (superposed model filename, min UB-LYS distance, num_clash).

    Jofily P, Kalyaanamoorthy S. P4ward: An Automated Modeling Platform for Protac Ternary Complexes. J Chem Inf Model. 2025 Aug 25;65(16):8806-8818. doi: 10.1021/acs.jcim.5c00614. Epub 2025 Aug 13. PMID: 40801829.
    '''
    ypath = Path(yaml_path).absolute()
    scoring_items = ('Fail_to_calculate',float('inf'),float('inf'))
    logging.info(f'Calculating superposing sc items: {ypath.stem}')
    predicted_structure_cif = ypath.parent / f'boltz_results_{ypath.stem}' / 'predictions' / ypath.stem / f'{ypath.stem}_model_0.cif'
    if not os.path.exists(predicted_structure_cif):
        return ('no_3d_cif_file',float('inf'),float('inf'))
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
                return ('no_surf_lys',float('inf'),float('inf'))
        ref_structure_dir = Path(__file__).parent.parent.parent / 'data/crl_models/' / (
            'crbn/' if 'crbn' in str(predicted_structure_cif) else 'vhl/')
        min_dist=float('inf')
        min_ref_file=''
        num_clash=float('inf')
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
                dist = min(
                    cmd.get_distance('ubc', f'chain P and resn LYS and name NZ and resi {resi}')
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
                    if generate_bestcif:
                        cmd.save(ypath.parent/f'{ypath.stem}_best.cif','ref')
            except Exception as e:
                logging.warning(f'Fail to treat {ypath.stem} superposing to {ref_file}')
                logging.warning(e)
                continue
        return (min_ref_file,min_dist,num_clash)
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
    extract_from_precomputed:bool=False,
    is_covalent_poi:bool=False,
    lig_poi_part:str='N(c1c(c(cc(c1))C)Nc1nc2c(cn1)cc(cc2)c1c(Cl)c(OC)cc(OC)c1Cl)C(=O)CC',
    lig_warhead_smarts: str='[C:1]-C-C=O', 
    covalent_resid_can: int= 104, 
    covalent_res_atomname: str='SG',
    generate_bestcif:bool=True,
    scaler=scaler,
)-> tuple[float,dict[str,float|str]]:
    '''Calculate a * lgKD_ternary + b * dist_ub_square + c * num_clash + bias. 
    '''
    inchikey=MolToInchiKey(MolFromSmiles(lig_smi))
    if not seq_e3:
        if name_e3=='vhl':
            seq_e3=SEQ_VHL
        elif name_e3=='crbn':
            seq_e3=SEQ_CRBN
        else:
            logging.error(f'Sequence of the given e3 {name_e3} is not given and e3 is not crbn or vhl. Will lead to failure.')
    if not pocket_e3:
        if name_e3=='vhl':
            pocket_e3=POCKET_VHL
        elif name_e3=='crbn':
            pocket_e3=POCKET_CRBN
        else:
            logging.error(f'Pocket of the given e3 {name_e3} is not given and e3 is not crbn or vhl. Will lead to failure.')
    if not os.path.exists(output_path):
        os.makedirs(output_path,exist_ok=True)
    msa_poi = os.path.join(msa_path, name_poi + '.a3m')
    msa_e3 = os.path.join(msa_path, name_e3 + '.a3m')
    yaml_ternary = os.path.join(
        output_path, f'{name_poi}_{name_e3}_{inchikey}_ternary.yaml')
    generate_boltz_covalent_or_not_ternary_yaml(lig_smi, seq_poi, pocket_poi, msa_poi, seq_e3,
                                pocket_e3, msa_e3, yaml_ternary,is_covalent_poi,lig_poi_part,lig_warhead_smarts,covalent_resid_can,covalent_res_atomname)
    lgKd_ternary=run_boltz_lgKd_prediction(yaml_ternary,extract_from_precomputed)
    super_scores=run_superpose_dist_ub_calc(yaml_ternary,generate_bestcif)
    dist_ub_square=super_scores[1]**2
    num_clash=super_scores[2]
    if dist_ub_square>10000:
        logging.warning(f'dist_ub is bigger than 100, the reference structure / fail reason is {super_scores[0]}')
    if np.any(np.isinf([[lgKd_ternary,dist_ub_square,num_clash]])):
        score=float('-inf')
    else:
        score=float(3/((1/(1-scaler.transform([[lgKd_ternary,dist_ub_square,num_clash]]))).sum(axis=1)))
    logging.info(f'lgKd_ternary: {lgKd_ternary:.3f},dist_ub_square: {dist_ub_square:.3f}, num_clash: {num_clash}, boltz weighted score: {score:.3f}')
    return score,{'lgKd_ternary':lgKd_ternary,'dist_ub_square':dist_ub_square,'dist_ub':super_scores[1],'num_clash':num_clash,'ref_model':super_scores[0]}

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
        return ComponentResults([scores],metadata=metadata)