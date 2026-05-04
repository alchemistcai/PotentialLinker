import logging
import os
import shutil
from boltz.data.msa.mmseqs2 import run_mmseqs2
from rdkit.Chem import MolFromSmiles,MolToInchiKey
from DockQ.DockQ import load_PDB, run_on_all_native_interfaces
def smi2inchikey(smiles:str) -> str|None:
    try:
        mol=MolFromSmiles(smiles)
        return MolToInchiKey(mol)
    except:return None

def clean_a3m(a3m_file:str,outfilename:str) -> None:
    with open(a3m_file,'r') as fin,open(outfilename,'w') as fout:
        for line in fin:
            line = line.replace("\x00", "")
            fout.write(line)

def get_a3m(seq:str,file_prefix:str) -> None:
    '''Get MSAs of `seq` from the server and save them into `file_prefix`.'''
    tmp_filepath=f'{file_prefix}_tmp.a3m'
    if not os.path.exists(f'{file_prefix}.a3m'):
        run_mmseqs2(seq,prefix=file_prefix)
        msa_path=f'{file_prefix}.a3m'
        shutil.move(f'{file_prefix}_env/bfd.mgnify30.metaeuk30.smag30.a3m',tmp_filepath)
        clean_a3m(tmp_filepath,msa_path)
    try:
        os.remove(tmp_filepath)
        shutil.rmtree(f'{file_prefix}_env',True)
    except Exception as e:
        logging.error(e)

def dockq_for_protac(native_cif:str,model_cif:str) -> dict[str, float]:
    '''Calculate dockq between native and model cif.
    Hardcode chain map to PEL:PEL in this repo.'''
    model=load_PDB(model_cif,small_molecule=True)
    native=load_PDB(native_cif,small_molecule=True)
    chain_map= {'P':'P','E':'E','L':'L',}
    results,total_dockq=run_on_all_native_interfaces(model,native,chain_map=chain_map)
    # global_dockq=total_dockq/len(results)
    # For small ligands, max(dockq)=2/3, so global_dockq is not suitable for PROTACs 
    protac_results={}
    protac_results['PE_DockQ']=results['PE']['DockQ']
    protac_results['PE_iRMSD']=results['PE']['iRMSD']
    protac_results['PE_LRMSD']=results['PE']['LRMSD']
    protac_results['PE_fnat']=results['PE']['fnat']
    protac_results['PL_LRMSD']=results['PL']['LRMSD']
    protac_results['EL_LRMSD']=results['EL']['LRMSD']
    return protac_results

