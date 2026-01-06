import logging
import os
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBIO import Select
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.Structure import Structure
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue
from Bio.PDB.Atom import Atom
from Bio.PDB.mmcifio import MMCIFIO



class CombineSelect(Select):
    def __init__(self,*selects:Select,operator=any) -> None:
        super().__init__()
        self.selects=selects
        self.operator=operator

    def accept_atom(self, atom):
        return self.operator(select.accept_atom(atom) for select in self.selects)
    def accept_residue(self, residue):
        return self.operator(select.accept_residue(residue) for select in self.selects)
    def accept_chain(self, chain):
        return self.operator(select.accept_chain(chain) for select in self.selects)
    def accept_model(self, model):
        return self.operator(select.accept_model(model) for select in self.selects)

class HetatmSelect(Select):
    def __init__(self,lig_chain:str='',lig_resname:str='') -> None:
        super().__init__()
        self.lig_chain=lig_chain
        self.lig_resname=lig_resname
    def accept_chain(self, chain:Chain):
        if self.lig_chain:
            return chain.id==self.lig_chain
        return True
    def accept_residue(self, residue:Residue):
        if self.lig_resname:
            return residue.get_resname()==self.lig_resname
        return residue.id[0].strip().startswith('H_')

class POISelect(Select):
    def __init__(self,poi_chain:str='') -> None:
        super().__init__()
        self.poi_chain=poi_chain
    
    def accept_chain(self, chain:Chain):
        if self.poi_chain:
            return chain.id==self.poi_chain
        return True
    
    def accept_residue(self, residue):
        return not residue.id[0].strip() # remove hetatm/water
    

def clean_structure(pdbid:str,ligid:str,poi_chain:str,lig_chain:str,workdir:str='data/PDB'):
    io=MMCIFIO()
    st_parser=MMCIFParser(auth_chains=True,auth_residues=True,QUIET=True)
    st=st_parser.get_structure(pdbid,os.path.join(workdir,pdbid+'.cif'))
    mmcif_dict=st_parser._mmcif_dict
    seq_offset_label_minus_auth=0
    for seq_id,pdb_seq_id,pdb_strand_id in zip(mmcif_dict['_pdbx_poly_seq_scheme.seq_id'],mmcif_dict['_pdbx_poly_seq_scheme.pdb_seq_num'],mmcif_dict['_pdbx_poly_seq_scheme.pdb_strand_id']):
        if pdb_strand_id == poi_chain and seq_id=='1':
            seq_offset_label_minus_auth=1-int(pdb_seq_id)
            break
    for chains,seq in zip(mmcif_dict['_entity_poly.pdbx_strand_id'],mmcif_dict['_entity_poly.pdbx_seq_one_letter_code_can']):
        if poi_chain in chains:
            seq_canon=seq.replace('\n','')
            break
    io.set_structure(st)
    io.save(os.path.join(workdir,f'{pdbid}_ref.cif'),CombineSelect(POISelect(poi_chain),HetatmSelect(lig_chain,ligid)))
    return seq_canon,seq_offset_label_minus_auth

def get_hetatm_residue(st:Structure,resname:str):
    for residue in st.get_residues():
        if residue.get_resname()==resname:
            # assume that st only has 1 hetatm residue
            return residue

def search_clean_pocket(st:Structure,lig_resname:str,seq_offset_label_minus_auth:int=0,radius:float|None=None):
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
    pocket_residues=[res for res in pocket_residues if res != hetatm_res][:3]
    # choose only 3 pocket residues.
    pocket_boltz_resids = [res.id[1]+seq_offset_label_minus_auth for res in pocket_residues]
    return pocket_boltz_resids

def get_pocket_info_from_pdb_info(pdbid:str,ligid:str,chain_poi:str,chain_lig:str,workdir:str='data/PDB',radius:float|None=None):
    logging.info(f'Processing {pdbid}')
    pdbid=pdbid.lower()
    seq_canon,seq_offset_label_minus_auth=clean_structure(pdbid,ligid,chain_poi,chain_lig,workdir)
    st_parser=MMCIFParser(auth_chains=True,auth_residues=True,QUIET=True)
    st=st_parser.get_structure(pdbid,os.path.join(workdir,f'{pdbid}_ref.cif'))
    pocket_boltz_resids=search_clean_pocket(st,ligid,seq_offset_label_minus_auth,radius)
    return seq_canon,pocket_boltz_resids


