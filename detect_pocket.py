import logging
import os
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.Structure import Structure
from Bio.PDB.Residue import Residue
import urllib.request
import gemmi
from rdkit import Chem
from rdkit.Chem import rdFMCS
from pymol import cmd

CCD_MIRROR="https://www.ebi.ac.uk/pdbe/static/files/pdbechem_v2/{0}.cif"
def download_ccd_file(code: str, save_dir: str = "ccd_files", mirror: str = CCD_MIRROR) -> str:
    os.makedirs(save_dir, exist_ok=True)
    local_path = os.path.join(save_dir, f"{code}.cif")
    if os.path.exists(local_path):
        return local_path
    url = mirror.format(code)
    try:
        urllib.request.urlretrieve(url, local_path)
        return local_path
    except Exception as e:
        raise FileNotFoundError(f"Fail to download CCD file {code}: {e}")

def parse_ccd2mol_and_smi(cif_path: str)->tuple[Chem.Mol|None,str]:
    """Parse CCD cif by gemmi, and convert it to a rdkit Mol with SMILES."""
    ccd_smi=''
    try:
        doc = gemmi.cif.read_file(cif_path)
        block = doc.sole_block()
        atoms = []
        bonds = []
        atom_loop = block.find('_pdbe_chem_comp_atom_depiction.', ['atom_id', 'element'])
        if not atom_loop:
            raise ValueError(f"No _chem_comp_atom table in ccd file: {cif_path}")
        for atom_row in atom_loop:
            atom_id = atom_row[0]
            element = atom_row[1]
            if element:
                atoms.append((atom_id, element))
        bond_loop = block.find('_chem_comp_bond.', ['atom_id_1', 'atom_id_2', 'value_order'])
        for bond_row in bond_loop:
            atom1, atom2 = bond_row[0], bond_row[1]
            bond_type = bond_row[2]
            bonds.append((atom1, atom2, bond_type))
        smi_loop= block.find('_pdbx_chem_comp_descriptor.',['comp_id','type','descriptor'])
        for smi_row in smi_loop:
            if smi_row[1]=='SMILES_CANONICAL':
                ccd_smi=smi_row[2].replace('"','')
                break
        if not atoms:
            raise ValueError(f"No atoms in cif file: {cif_path}")
        mol = Chem.RWMol()
        atom_idx_map = {}
        for atom_id, element in atoms:
            atom_id=atom_id.strip()
            atomic_num = Chem.GetPeriodicTable().GetAtomicNumber(element)
            if atomic_num == 0:
                continue
            atom = Chem.Atom(atomic_num)
            atom.SetProp("_Name", atom_id)
            idx = mol.AddAtom(atom)
            atom_idx_map[atom_id] = idx
        bond_type_map = {
            'SING': Chem.BondType.SINGLE,
            'DOUB': Chem.BondType.DOUBLE,
            'TRIP': Chem.BondType.TRIPLE,
            'AROM': Chem.BondType.AROMATIC
        }
        for atom1, atom2, btype in bonds:
            if atom1 in atom_idx_map and atom2 in atom_idx_map:
                bt = bond_type_map.get(btype, Chem.BondType.SINGLE)
                mol.AddBond(atom_idx_map[atom1], atom_idx_map[atom2], bt)
        mol = mol.GetMol()
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL)
        return mol,ccd_smi
    except Exception as e:
        logging.error(f"Fail to parse ccd file: {e}")
        return None,ccd_smi
def get_atomname_map_ccd2boltz_with_smi(
    ccd_code: str,
    ccd_dir: str = "ccd_files",
    ccd_mirror: str = CCD_MIRROR,
    only_get_smi:bool=False
) -> tuple[dict[str, str],str]:
    """
    For a given ccd code, download its cif file, generate atom name map{cif_name: boltz_name} and its SMILES.
    """
    
    def parse_boltz_mol(smi:str) -> Chem.Mol|None:
        """
        """
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        Chem.SanitizeMol(mol)
        canonical_order = Chem.CanonicalRankAtoms(mol)
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        for atom, can_idx in zip(mol.GetAtoms(), canonical_order):
            atom_name = (atom.GetSymbol().upper() + str(can_idx + 1)).strip()
            if len(atom_name) > 4:
                raise ValueError(f"Atom name is longer than 4: {atom_name}")
            atom.SetProp("_boltz_name", atom_name)
        return mol
    
    def perform_mcs_atom_name_mapping(
        mol_ccd: Chem.Mol,
        mol_boltz: Chem.Mol
    ) -> dict[str, str]|None:
        if mol_ccd is None or mol_boltz is None:
            return None
        params = rdFMCS.MCSParameters()
        params.AtomTyper = rdFMCS.AtomCompare.CompareElements
        params.BondTyper = rdFMCS.BondCompare.CompareOrder
        params.AtomCompareParameters.RingMatchesRingOnly = True
        params.BondCompareParameters.RingMatchesRingOnly = True
        params.MaximizeBonds = True
        params.Timeout = 2
        mcs_result = rdFMCS.FindMCS([mol_ccd, mol_boltz], params)
        if not mcs_result.smartsString:
            logging.warning("MCS match failed.")
            return None
        mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
        match_ccd = mol_ccd.GetSubstructMatch(mcs_mol)
        match_boltz = mol_boltz.GetSubstructMatch(mcs_mol)
        if not match_ccd or not match_boltz:
            logging.warning("Fail to match CCD mol and boltz mol.")
            return None
        idx_map = {ccd_idx: boltz_idx for ccd_idx, boltz_idx in zip(match_ccd, match_boltz)}
        atom_name_map = {}
        for ccd_idx, boltz_idx in idx_map.items():
            ccd_atom = mol_ccd.GetAtomWithIdx(ccd_idx)
            boltz_atom = mol_boltz.GetAtomWithIdx(boltz_idx)
            ccd_name = ccd_atom.GetProp("_Name") if ccd_atom.HasProp("_Name") else f"UNK{ccd_idx}"
            boltz_name = boltz_atom.GetProp("_boltz_name") if boltz_atom.HasProp("_boltz_name") else f"BOLTZ{boltz_idx}"
            atom_name_map[ccd_name] = boltz_name
        return atom_name_map
    
    cif_path = download_ccd_file(ccd_code, ccd_dir, ccd_mirror)
    mol_ccd,smi_ccd = parse_ccd2mol_and_smi(cif_path)
    if only_get_smi:
        return {},smi_ccd
    if mol_ccd is None:
        raise ValueError(f"Unable to parse CCD cif file: {cif_path}")
    mol_boltz = parse_boltz_mol(smi_ccd)
    if mol_boltz is None:
        raise ValueError(f"Unable to create Boltz mol: {ccd_code}")
    atomname_map = perform_mcs_atom_name_mapping(mol_ccd, mol_boltz)
    if atomname_map is None:
        raise ValueError(f"MCS match failed: {ccd_code}")
    return atomname_map,smi_ccd

def clean_structure(pdbid:str,ligid:str,chain_poi:str,chain_lig:str,workdir:str='data/PDB',chain_e3:str='',canon_atomname:bool=False):
    st_parser=MMCIFParser(auth_chains=True,auth_residues=True,QUIET=True)
    st_parser.get_structure(pdbid,os.path.join(workdir,f'{pdbid}.cif'))
    mmcif_dict=st_parser._mmcif_dict
    seq_auth2can_poi={}
    seq_auth2can_e3={}
    for seq_id,pdb_seq_id,pdb_strand_id in zip(mmcif_dict['_pdbx_poly_seq_scheme.seq_id'],mmcif_dict['_pdbx_poly_seq_scheme.pdb_seq_num'],mmcif_dict['_pdbx_poly_seq_scheme.pdb_strand_id']):
        if pdb_strand_id == chain_poi:
            seq_auth2can_poi[int(pdb_seq_id)]=int(seq_id)
        if chain_e3 and pdb_strand_id == chain_e3:
            seq_auth2can_e3[int(pdb_seq_id)]=int(seq_id)
    for chains,seq in zip(mmcif_dict['_entity_poly.pdbx_strand_id'],mmcif_dict['_entity_poly.pdbx_seq_one_letter_code_can']):
        if chain_poi in chains:
            seq_poi_canon=seq.replace('\n','')
        if chain_e3 and chain_e3 in chains:
            seq_e3_canon=seq.replace('\n','')
    os.makedirs(os.path.join(workdir,'clean'), exist_ok=True)
    cmd.reinitialize()
    cmd.load(os.path.join(workdir,pdbid+'.cif'),'ref')
    cmd.select('lig',f'chain {chain_lig} and resn {ligid}')
    cmd.select('poi',f'chain {chain_poi} and not hetatm')
    cmd.alter('lig','chain="L"')
    cmd.alter('lig','segi="L"')
    cmd.alter('poi','chain="P"')
    cmd.alter('poi','segi="P"')
    if chain_e3:
        cmd.select('e3',f'chain {chain_e3} and not hetatm')
        cmd.alter('e3','chain="E"')
        cmd.alter('e3','segi="E"')
    atomname_ccd2boltz,smi=get_atomname_map_ccd2boltz_with_smi(ligid,workdir,only_get_smi=(not canon_atomname))
    if canon_atomname:
        cmd.alter(f'lig',f'name+="_T"') # avoid possible same name 
        for ccd_name,boltz_name in atomname_ccd2boltz.items():
            cmd.alter(f'lig and name {ccd_name}_T',f'name="{boltz_name}"')
    cmd.save(os.path.join(workdir,'clean',f'{pdbid}_ref.cif'),'poi or lig'+(' or e3' if chain_e3 else ''))
    return (seq_poi_canon,seq_auth2can_poi,smi)+((seq_e3_canon,seq_auth2can_e3) if chain_e3 else ())+((atomname_ccd2boltz,) if canon_atomname else ())

def get_hetatm_residue(st:Structure,resname:str):
    for residue in st.get_residues():
        if residue.get_resname()==resname:
            # assume that st only has 1 hetatm residue
            return residue

def search_clean_pocket(st:Structure,lig_resname:str,seq_auth2can:dict[int,int],radius:float=0) -> list[int]:
    ns=NeighborSearch(list(st.get_atoms()))
    hetatm_res=get_hetatm_residue(st,lig_resname)
    pocket_residues=set()
    if radius:
        for hetatm in hetatm_res:
            pocket_residues.update(ns.search(hetatm.coord,radius,level='R'))
    else:
        radius=0.5
        while True:
            for hetatm in hetatm_res:
                pocket_residues.update(ns.search(hetatm.coord,radius,level='R'))
                radius+=0.2
            if len(pocket_residues)>=4:
                break
    pocket_residues=[res for res in pocket_residues if res != hetatm_res]
    pocket_boltz_resids = [seq_auth2can[res.id[1]] for res in pocket_residues]
    return pocket_boltz_resids

def search_clean_ppi(st:Structure,chain_poi:str,chain_lig:str,chain_e3:str,seq_auto2can_poi:dict[int,int],seq_auto2can_e3:dict[int,int]):
    poi_resis:set[Residue] = set()
    e3_resis:set[Residue]=set()
    found_lig=False
    for res in st.get_residues():
        chain_id=res.get_parent().id
        if chain_id==chain_poi:
            poi_resis.add(res)
        elif chain_id==chain_e3:
            e3_resis.add(res)
        elif not found_lig and chain_id==chain_lig:
            lig_res=res
            found_lig=True
    ns=NeighborSearch(list(st.get_atoms()))
    contact_pairs=set()
    for poi_res in poi_resis:
        for atom in poi_res:
            ppi_e3_resis=set(ns.search(atom.coord,6,level='R')) & e3_resis
            # 4A for PPI may be too sparse for certein PROTACs, like 6w8i
            for e3_res in ppi_e3_resis:
                contact_pairs.add((poi_res.id[1],e3_res.id[1]))
    ppi = [(seq_auto2can_poi[poi_resi],seq_auto2can_e3[e3_resi]) for poi_resi,e3_resi in contact_pairs]

    pocket_resis=set()
    for atom in lig_res:
        pocket_resis.update(ns.search(atom.coord,4,level='R'))
    pocket_poi=pocket_resis&poi_resis
    pocket_e3=pocket_resis&e3_resis
    return ppi,[seq_auto2can_poi[poi_res.id[1]] for poi_res in pocket_poi],[seq_auto2can_e3[e3_res.id[1]] for e3_res in pocket_e3]

def get_pocket_info_from_pdb_info(pdbid:str,ligid:str,chain_poi:str,chain_lig:str,workdir:str='data/PDB',radius:float=0,canon_atomname:bool=False):
    '''Get pocket residues from cif structure. Will remove unwanted hetatm and chains.'''
    logging.info(f'Processing {pdbid}')
    pdbid=pdbid.lower()
    seq_canon,seq_auth2can,*_ =clean_structure(pdbid,ligid,chain_poi,chain_lig,workdir,canon_atomname=canon_atomname)
    st_parser=MMCIFParser(auth_chains=True,auth_residues=True,QUIET=True)
    st=st_parser.get_structure(pdbid,os.path.join(workdir,'clean',f'{pdbid}_ref.cif'))
    pocket_boltz_resids=search_clean_pocket(st,ligid,seq_auth2can,radius)
    return seq_canon,pocket_boltz_resids

def get_ppi_info_from_pdb_info(pdbid:str,ligid:str,chain_poi:str,chain_lig:str,chain_e3:str,workdir:str='data/PDB'):
    '''Get ppi residues from cif structure. Will remove unwanted hetatm and chains.'''
    logging.info(f'Processing {pdbid}')
    st_parser=MMCIFParser(auth_chains=True,auth_residues=True,QUIET=True)
    pdbid=pdbid.lower()
    seq_poi,seq_auth2can_poi,smi,seq_e3,seq_auth2can_e3,atomname_ccd2boltz=clean_structure(pdbid,ligid,chain_poi,chain_lig,workdir,chain_e3=chain_e3,canon_atomname=True)
    st=st_parser.get_structure(pdbid,os.path.join(workdir,'clean',f'{pdbid}_ref.cif'))
    ppi,pocket_poi,pocket_e3=search_clean_ppi(st,'P','L','E',seq_auth2can_poi,seq_auth2can_e3)
    return seq_poi,seq_e3,seq_auth2can_poi,seq_auth2can_e3,smi,ppi,pocket_poi,pocket_e3,atomname_ccd2boltz
