import io
import json
import logging
import os
from pathlib import Path
import random
from pymol import cmd
from Bio.PDB import  PDBParser, NeighborSearch
from rdkit.Chem import Mol,MolFromSmiles,AddHs,MolFromSmarts,AssignStereochemistry,AllChem

# pdbid: 8rqa, CRBN-midi with shorter sequence
POCKET_CRBN = [283, 289, 282, 255, 254, 281, 305, 303, 256]
SEQ_CRBN = 'SAKKPNIINFDTSLPTSHTYLGADMEEFHGRTLHDDDSIQVIPVLPQVMMILVPGQTLPLQLFHPQEVSMVRNLIQNDRTFAVLAYSNVQEREAEFGTTAEIYAYREEQDFGIEIVKVKAIGRQRFKVLELRTQSDGIQQAKVQILPEGSGDAETLMDRIKKQLREWDENLKDDSLPSNPIDFSYWVAANLPIDDSLRIQLLKIDSAIQRLRCELDIMNKCTSLCCKQCQETEITTKNEIFSLSREGPMAAYVNPHGYVHEILTVYKACNLNLIGRPSTEHSWFPGYAWTVAQCKICASHIGWKFTATKKDMSPQKFWGLTRSALIPTI'

# pdbid: 6GFY
POCKET_VHL = [60, 61, 66, 37, 64, 58, 40, 59, 48, 47]
SEQ_VHL = 'GSMEAGRPRPVLRSVNSREPSQVIFCNRSPRVVLPVWLNFDGEPQPYPTLPPGTGRRIHSYRGHLWLFRDAGTHDGLLVNQTELFVPSLNVDGQPIFANITLPVYTLKERCLQVVRSLVKPENYRRLDIVRSLYEDLEDHPNVQKDLERLTQERIAHQRMGD'

IntOrInf=int|float


def find_warhead_atom_name(lig_mol: Mol,lig_part_smiles: str,warhead_smarts: str,part_label: str) -> str|None:
    '''Return the boltz atom name (e.g. `C12`) of the warhead atom that belongs to `lig_part_smiles`.

    `lig_mol` must carry explicit hydrogens. Return None and log the reason when the part is not a
    valid SMILES, the part is missing from `lig_mol`, the warhead SMARTS matches nothing, or no
    warhead match sits fully inside the given part.
    '''
    part_mol=MolFromSmiles(lig_part_smiles)
    if part_mol is None:
        logging.error(f'Given {part_label} part substructure {lig_part_smiles} is not a valid smiles.')
        return None
    part_match=lig_mol.GetSubstructMatch(part_mol)
    if not part_match:
        logging.error(f'Given {part_label} part substructure {lig_part_smiles} not found.')
        return None
    pattern=MolFromSmarts(warhead_smarts)
    if pattern is None:
        logging.error(f'Given {part_label} covalent warhead SMARTS {warhead_smarts} is not valid.')
        return None
    all_matches=lig_mol.GetSubstructMatches(pattern)
    if not all_matches:
        logging.error(f'Given {part_label} covalent warhead SMARTS {warhead_smarts} not found.')
        return None
    part_atom_ids=set(part_match)
    part_matches=[match for match in all_matches if all(idx in part_atom_ids for idx in match)]
    if not part_matches:
        logging.error(f'Given {part_label} covalent warhead SMARTS {warhead_smarts} not in {part_label} part substructure {lig_part_smiles}.')
        return None
    target_atom_idx=random.choice(part_matches)[0]
    canonical_order=AllChem.CanonicalRankAtoms(lig_mol) # get atom name according to boltz schema
    AssignStereochemistry(lig_mol, force=True, cleanIt=True)
    for atom, can_idx in zip(lig_mol.GetAtoms(), canonical_order):
        if atom.GetIdx()==target_atom_idx:
            return atom.GetSymbol().upper() + str(can_idx + 1)
    return None


def generate_boltz_ternary_yaml(
    lig_smi: str,
    seq_poi: str,
    pocket_poi: list[int],
    msa_poi: str,
    seq_e3: str,
    pocket_e3: list[int],
    msa_e3: str,
    output_filename: str,
    is_covalent_poi: bool=False,
    lig_poi_part:str='N(c1c(c(cc(c1))C)Nc1nc2c(cn1)cc(cc2)c1c(Cl)c(OC)cc(OC)c1Cl)C(=O)CC', # FGFR4 BLU9931
    lig_warhead_smarts: str='[C:1]-C-C=O', #  reaction site after addition
    covalent_resid_can: int= 104, # (CYS 552) - offset 448
    covalent_res_atomname: str='SG',
    contact_ppi: list[tuple[int, int]] | None = None,
    template_cif_or_path:str='',
    template_chain_poi_can:str='',
    template_chain_e3_can:str='',
    is_covalent_e3: bool=False,
    lig_e3_part: str='',
    lig_e3_warhead_smarts: str='',
    covalent_resid_e3_can: int=0,
    covalent_res_atomname_e3: str='SG',
):
    pocket_restraint=''
    if pocket_poi or pocket_e3:
        pocket_poi_input = ','.join(f'[P, {resid}]' for resid in pocket_poi)
        pocket_e3_input = ','.join(f'[E, {resid}]' for resid in pocket_e3)
        pocket_restraint=f"""  - pocket:
      binder: L
      contacts: [ {pocket_poi_input} {',' if pocket_poi and pocket_e3 else ''} {pocket_e3_input} ]
      max_distance: 4
      force: true"""+'\n'
    ppi_restraint = ''
    if contact_ppi:
        ppi_restraint = '\n'.join(f'''  - contact:
      token1: [P, {ppi[0]}]
      token2: [E, {ppi[1]}]
      max_distance: 6
      force: true''' for ppi in contact_ppi) + '\n'
    template_restraint=''
    if template_cif_or_path:     
        template_restraint=f"""templates:
    - cif: {os.path.abspath(template_cif_or_path)}
      chain_id: [P, E]
      template_id: [{template_chain_poi_can}, {template_chain_e3_can}]
      force: true
      threshold: 1"""+'\n'
    # trying to parse pymol cif leads to bad RMSD and awkward geometry when the template files miss internal residues 
    covalent_constraint=''
    e3_covalent_constraint=''
    if is_covalent_poi or is_covalent_e3:
        tmp_mol=MolFromSmiles(lig_smi)
        if tmp_mol is None:
            logging.error(f'Given lig_smi {lig_smi} is not a valid smiles.')
            return None
        tmp_mol = AddHs(tmp_mol)
        if is_covalent_poi:
            target_atom_name=find_warhead_atom_name(tmp_mol,lig_poi_part,lig_warhead_smarts,'Poi')
            if target_atom_name is None:
                return None
            covalent_constraint=f"""  - bond:
      atom1: [L, 1, {target_atom_name}]
      atom2: [P, {covalent_resid_can}, {covalent_res_atomname}]"""+'\n'
        if is_covalent_e3:
            if not (lig_e3_part and lig_e3_warhead_smarts and covalent_resid_e3_can):
                logging.warning('is_covalent_e3 is set but lig_e3_part / lig_e3_warhead_smarts / covalent_resid_e3_can is missing; the E3 covalent bond is skipped.')
            else:
                target_atom_name=find_warhead_atom_name(tmp_mol,lig_e3_part,lig_e3_warhead_smarts,'E3')
                if target_atom_name is None:
                    return None
                e3_covalent_constraint=f"""  - bond:
      atom1: [L, 1, {target_atom_name}]
      atom2: [E, {covalent_resid_e3_can}, {covalent_res_atomname_e3}]"""+'\n'
    constraint_header='constraints:\n' if any((pocket_restraint,ppi_restraint,template_restraint,covalent_constraint,e3_covalent_constraint)) else ''
    e3_bond_doc='  # The E3 covalent bond is optional: pass is_covalent_e3, lig_e3_part, lig_e3_warhead_smarts and\n  # covalent_resid_e3_can to emit it; it is omitted when they are left empty.\n' if constraint_header else ''
    yaml_config = f'''sequences:
  - protein:
      id: [P]
      sequence: {seq_poi}
      msa: {os.path.abspath(msa_poi) if seq_poi!=seq_e3 else os.path.abspath(msa_e3)}
  - protein:
      id: [E]
      sequence: {seq_e3}
      msa: {os.path.abspath(msa_e3)}
  - ligand:
      id: [L]
      smiles: '{lig_smi}'
{constraint_header}{e3_bond_doc}{covalent_constraint}{e3_covalent_constraint}{pocket_restraint}{ppi_restraint}{template_restraint}properties:
  - affinity:
      binder: L
'''

    with open(output_filename, 'w') as f:
        f.write(yaml_config)


def generate_boltz_poi_yaml(lig_smi: str, seq_poi: str, pocket_poi: list[int],
                            msa_poi: str, output_filename: str):
    pocket_poi_input = ','.join(f'[P, {resid}]' for resid in pocket_poi)
    template = f'''sequences:
  - protein:
      id: [P]
      sequence: {seq_poi}
      msa: {os.path.abspath(msa_poi)}
  - ligand:
      id: [L]
      smiles: '{lig_smi}'
constraints:
  - pocket:
      binder: L
      contacts: [ {pocket_poi_input} ]
      max_distance: 4
      force: true
properties:
    - affinity:
        binder: L
'''
    with open(output_filename, 'w') as f:
        f.write(template)


def generate_boltz_e3_yaml(lig_smi: str, seq_e3: str, pocket_e3: list[int],
                           msa_e3: str, output_filename: str):
    pocket_poi_input = ','.join(f'[E, {resid}]' for resid in pocket_e3)
    template = f'''sequences:
  - protein:
      id: [E]
      sequence: {seq_e3}
      msa: {os.path.abspath(msa_e3)}
  - ligand:
      id: [L]
      smiles: '{lig_smi}'
constraints:
  - pocket:
      binder: L
      contacts: [ {pocket_poi_input} ]
      max_distance: 4
      force: true
properties:
    - affinity:
        binder: L
'''
    with open(output_filename, 'w') as f:
        f.write(template)


def run_boltz_lgKd_prediction(yaml_path: str,extract_from_precomputed:bool=False) -> float:
    '''Return boltz2 affinity prediciton (lgKd \\mu M).
    
    yaml_path is the yaml filename with optional leading directory.
    
    When extract_from_precomputed is True, will not use boltz to predict aff. Parse affinity from precomputed json files.'''
    logging.info(f'Predicting {yaml_path}')
    ypath = Path(yaml_path)
    result_json_path = ypath.parent / f'boltz_results_{ypath.stem}' / 'predictions' / ypath.stem / f'affinity_{ypath.stem}.json'
    try:
        if not extract_from_precomputed and not os.path.exists(result_json_path):
            os.system(f'boltz predict --seed 0 --use_msa_server --use_potentials --affinity_mw_correction {ypath} --out_dir {ypath.parent}')
    except Exception as e:
        logging.warning(e)
    try:
        with open(result_json_path, 'r') as f:
            return float(json.load(f)['affinity_pred_value'])
    except Exception as e:
        logging.warning(e)
        return float('inf')
    return float('inf')

def run_superpose_scoring_items_calc(yaml_path: str,generate_bestcif:bool=True,best_record_only:bool=False,name_e3_origin:str='',calc_bsa:bool=True) -> list[list]|list:
    '''Use P4ward generated CRL complex models to calculate 
    [[superposed model filename, min UB-LYS distance, has surface lys after Boltz docking, buried surface area,clash atoms,closest lys resid].

    If `best_record_only`, will return [superposed model filename, min UB-LYS distance, ...].

    If name_e3_origin is given, use your E2/Ub/E3 reference directory instead.

    If not `calc_bsa`, bsa and surface lys info will be not in results. 

    num_clash is defined by:
    Jofily P, Kalyaanamoorthy S. P4ward: An Automated Modeling Platform for Protac Ternary Complexes. J Chem Inf Model. 2025 Aug 25;65(16):8806-8818. doi: 10.1021/acs.jcim.5c00614. Epub 2025 Aug 13. PMID: 40801829.
    '''
    ypath = Path(yaml_path).absolute()
    min_dist=float('inf')
    num_clash=float('inf')
    bsa=0
    lys_poi_surf_resis_after_docking=0
    closest_lys_resid=-10000
    scoring_items = [['Fail_to_calculate',min_dist,num_clash,closest_lys_resid]]
    if calc_bsa:
        scoring_items[0].insert(2,lys_poi_surf_resis_after_docking)
        scoring_items[0].insert(3,bsa)
    logging.info(f'Calculating superposing sc items: {ypath.stem}')
    predicted_structure_cif = ypath.parent / f'boltz_results_{ypath.stem}' / 'predictions' / ypath.stem / f'{ypath.stem}_model_0.cif'
    if not os.path.exists(predicted_structure_cif):
        scoring_items[0][0]='no_3d_cif_file'
        return scoring_items[0] if best_record_only else scoring_items
    try:
        cmd.reinitialize()
        cmd.load(predicted_structure_cif, 'ternary')
        sasa_poi = cmd.get_area('ternary and chain P')
        lys_poi_sasa_info_after_docking = cmd.get_sasa_relative('chain P and resn LYS',
                                                  subsele='sidechain')
        lys_poi_surf_resis=lys_poi_surf_resis_after_docking = [
            idx_str for (*_, idx_str), sasa_rel in lys_poi_sasa_info_after_docking.items() if sasa_rel >= 0.4
        ]
        # dict[(objname,segi,chain,resi), sasa_rel]
        # relative side-chain SASA >= 40% is on surface (haddock surface cutoff).
        if not lys_poi_surf_resis_after_docking: # fallback to undocked Lys
        # suitable for flexible segments
            cmd.copy_to('poi','ternary and chain P')
            lys_poi_sasa_info_no_docking = cmd.get_sasa_relative('poi and chain P and resn LYS',subsele='sidechain')
            lys_poi_surf_resis=lys_poi_surf_reses_no_docking = [
                idx_str for (*_, idx_str), sasa_rel in lys_poi_sasa_info_no_docking.items()
            ]
            if not lys_poi_surf_reses_no_docking:
                logging.debug(str(lys_poi_sasa_info_no_docking))
                scoring_items[0][0]='no_surf_lys'
                return scoring_items[0] if best_record_only else scoring_items
        if name_e3_origin:
            ref_structure_dir = Path(__file__).parent.parent.parent / 'data/crl_models' /name_e3_origin
        elif 'crbn' in str(predicted_structure_cif):
            ref_structure_dir = Path(__file__).parent.parent.parent / 'data/crl_models' /'crbn/'
        elif 'vhl' in str(predicted_structure_cif):
            ref_structure_dir = Path(__file__).parent.parent.parent / 'data/crl_models' /'vhl/'
        if not ref_structure_dir.exists():
            scoring_items[0][0]='no_UB_E2_cif_file'
            return scoring_items[0] if best_record_only else scoring_items
        scoring_items_list=[]
        for ref_file in os.listdir(ref_structure_dir):
            dist = float('inf')
            bsa_after_superposing = 0
            clash = float('inf')
            lys_resid=-10000
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
                dist,lys_resid = min((
                    cmd.get_distance('ubc', f'chain P and resn LYS and name NZ and resi {resi}'),resi)
                    for resi in lys_poi_surf_resis)

                # bsa
                # very fast,so calculate it anyway
                cmd.remove('chain C')
                cmd.copy_to('ref', 'ternary')
                sasa_poi_and_e2e3ub = cmd.get_area('ref and chain P')
                bsa_after_superposing = sasa_poi - sasa_poi_and_e2e3ub

                # clash
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
                scoring_items_list.append(
                    [ref_file, dist, clash,int(lys_resid)])
                if calc_bsa:
                    scoring_items_list[-1].insert(2,int(bool(lys_poi_surf_resis_after_docking)))
                    scoring_items_list[-1].insert(3,bsa_after_superposing)
                if dist<min_dist:
                    min_dist=dist
                    num_clash=clash
                    closest_lys_resid=int(lys_resid)
                    if generate_bestcif:
                        cmd.save(ypath.parent/f'{ypath.stem}_best.cif','ref')
            except Exception as e:
                logging.warning(f'Fail to treat {ypath.stem} superposing to {ref_file}')
                logging.warning(e)
                continue
        return min(scoring_items_list,key=lambda row:row[1]) if best_record_only else scoring_items_list
    except Exception as e:
        logging.warning(f'Fail to treat {ypath.stem} calculating superposing scores.')
        logging.warning(e)
        return scoring_items[0] if best_record_only else scoring_items


def calculate_affinity_score(
    lig_smi: str,
    inchikey: str,
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
    only_ternary:bool=False
)-> list[tuple[str,str,str,float,float,float,str,float,int,float,IntOrInf]]:
    '''Calculate name_poi* ,name_e3*, inchikey*, lgKd(POI), lgKd(E3), lgKd(ternary),superpose ref_filename, min UB distance, has surface lys after docking, BSA, clash atoms. 

    * fields are primary keys.
    '''
    if not seq_e3:
        if name_e3=='vhl':
            seq_e3=SEQ_VHL
        elif name_e3=='crbn':
            seq_e3=SEQ_CRBN
        else:
            logging.error(f'Sequence of the given e3 {name_e3} is not given and e3 is not crbn or vhl. May lead to failure.')
    if pocket_e3 is None:
        if name_e3=='vhl':
            pocket_e3=POCKET_VHL
        elif name_e3=='crbn':
            pocket_e3=POCKET_CRBN
        else:
            logging.error(f'Pocket of the given e3 {name_e3} is not given and e3 is not crbn or vhl. May lead to failure.')
    if not os.path.exists(output_path):
        os.makedirs(output_path,exist_ok=True)
    msa_poi = os.path.join(msa_path, name_poi + '.a3m')
    msa_e3 = os.path.join(msa_path, name_e3 + '.a3m')
    yaml_ternary = os.path.join(
        output_path, f'{name_poi}_{name_e3}_{inchikey}_ternary.yaml')
    yaml_poi = os.path.join(output_path,
                            f'{name_poi}_{name_e3}_{inchikey}_poi.yaml')
    yaml_e3 = os.path.join(output_path,
                           f'{name_poi}_{name_e3}_{inchikey}_e3.yaml')
    generate_boltz_ternary_yaml(lig_smi, seq_poi, pocket_poi, msa_poi, seq_e3,
                                pocket_e3, msa_e3, yaml_ternary, contact_ppi=contact_ppi,template_cif_or_path=template_cif_or_path,template_chain_poi_can=template_chain_poi_can,template_chain_e3_can=template_chain_e3_can)
    lgKd_ternary=run_boltz_lgKd_prediction(yaml_ternary,extract_from_precomputed)
    if not only_ternary:
        generate_boltz_poi_yaml(lig_smi, seq_poi, pocket_poi, msa_poi, yaml_poi)
        generate_boltz_e3_yaml(lig_smi, seq_e3, pocket_e3, msa_e3, yaml_e3)
        lgKd_poi=run_boltz_lgKd_prediction(yaml_poi,extract_from_precomputed)
        lgKd_e3=run_boltz_lgKd_prediction(yaml_e3,extract_from_precomputed)
    else:
        lgKd_poi=lgKd_e3=0
    one_line_record=[name_poi,name_e3,inchikey,lgKd_poi,lgKd_e3,lgKd_ternary]
    scoring_items=run_superpose_scoring_items_calc(yaml_ternary)
    return [one_line_record+scs for scs in scoring_items]

