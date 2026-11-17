from ast import literal_eval
import warnings
from pymol import cmd
from collections import UserDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
import shutil
from typing import Iterable
import urllib.request
import json
import time
import yaml
import gemmi
from boltz.data.msa.mmseqs2 import run_mmseqs2
import pandas as pd
from rdkit.Chem import MolFromSmiles,MolToInchiKey
from DockQ.DockQ import load_PDB, run_on_all_native_interfaces
from concurrent.futures import ThreadPoolExecutor, as_completed
from Bio.Align import PairwiseAligner
warnings.filterwarnings(category=UserWarning,action='ignore')

EVD_GOLD=4
EVD_SILVER=3
EVD_BRONZE=2
EVD_OTHER=1
EVD_NONE=0

# offset added to a boltz resid that the alignment cannot map, so unmatched residues never collide with uniprot numbering
UNMAPPED_RESI_OFFSET=5000

# uniprot accession of the E3 ligases used by this repo, needed to calibrate chain E to uniprot numbering
E3_UNIPROT_ID={'crbn':'Q96SW2','vhl':'P40337'}

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
        print(1)
        run_mmseqs2(seq,prefix=file_prefix)
        msa_path=f'{file_prefix}.a3m'
        shutil.move(f'{file_prefix}_env/bfd.mgnify30.metaeuk30.smag30.a3m',tmp_filepath)
        clean_a3m(tmp_filepath,msa_path)
    try:
        os.remove(tmp_filepath)
        shutil.rmtree(f'{file_prefix}_env',True)
    except Exception as e:
        pass

def dockq_for_protac(native_cif:str,model_cif:str) -> dict[str, float]:
    '''Calculate dockq between native and model cif.
    Hardcode chain map to PEL:PEL in this repo.'''
    try:
        model=load_PDB(model_cif,small_molecule=True)
        native=load_PDB(native_cif,small_molecule=True)
        chain_map= {'P':'P','E':'E','L':'L',}
        results,total_dockq=run_on_all_native_interfaces(model,native,chain_map=chain_map)
        # global_dockq=total_dockq/len(results)
        # For small ligands, max(dockq)=2/3, min(dockq)=1/3, so global_dockq is not suitable for PROTACs 
        protac_results={}
        protac_results['PE_DockQ']=results['PE']['DockQ']
        protac_results['PE_iRMSD']=results['PE']['iRMSD']
        protac_results['PE_LRMSD']=results['PE']['LRMSD']
        protac_results['PE_fnat']=results['PE']['fnat']
        protac_results['PL_LRMSD']=results['PL']['LRMSD']
        protac_results['EL_LRMSD']=results['EL']['LRMSD']
    except:
        protac_results={}
        protac_results['PE_DockQ']=0
        protac_results['PE_iRMSD']=0
        protac_results['PE_LRMSD']=0
        protac_results['PE_fnat']=0
        protac_results['PL_LRMSD']=0
        protac_results['EL_LRMSD']=0
    return protac_results

class Idx2MaxScore(UserDict):
    '''A dict only accept the highest score.'''
    def __setitem__(self, key, value):
        if key in self:
            if value > self[key]:
                self.data[key] = value
        else:
            self.data[key] = value


def idxmap_boltz2uniprot(seq_boltz:str,seq_uni:str) -> dict[int,int]:
    '''Return a sequence alignment map between seq_boltz and seq_uni, both of which starts from 1.'''
    aligner = PairwiseAligner('blastp')
    alignments = aligner.align(seq_uni, seq_boltz)
    result = {}
    if not alignments:
        return result
    best_alignment = alignments[0]
    indices=best_alignment.inverse_indices
    
    for boltz_idx, uni_idx in zip(indices[0],indices[1]):
        uni_idx= uni_idx+1 if uni_idx!=-1 else -1
        boltz_idx= boltz_idx+1 if boltz_idx!=-1 else -1
        result[boltz_idx]= uni_idx
    return result   

@dataclass
class ExperimentConstraints:
    '''Experimental constraints expressed in the boltz chain numbering.

    Build it with `uniprot_to_boltz_constraints`; every attribute can be passed straight to
    `generate_boltz_ternary_yaml`, `calculate_one_affinity_score` or `generate_pymol_pse`.
    Only constraints that need the uniprot to boltz remap live here; warhead SMILES, SMARTS and
    bonding atom names go to the yaml generator directly.
    '''
    pocket_poi: list[int] = field(default_factory=list)
    pocket_e3: list[int] = field(default_factory=list)
    contact_ppi: list[tuple[int,int]] = field(default_factory=list)
    covalent_resid_can: int = 0
    covalent_resid_e3_can: int = 0


def uniprot_to_boltz_constraints(
    seq_poi_boltz:str,seq_poi_uniprot:str,
    pocket_poi_uniprot:Iterable[int]=(),
    covalent_resid_poi_uniprot:int=0,
    seq_e3_boltz:str='',seq_e3_uniprot:str='',
    pocket_e3_uniprot:Iterable[int]=(),
    covalent_resid_e3_uniprot:int=0,
    contact_ppi_uniprot:Iterable[tuple[int,int]]=(),
) -> ExperimentConstraints:
    '''Convert experimental uniprot residue numbers into the boltz numbering used by the yaml generator.

    All inputs are uniprot residue numbers.

    Residues in the alignment cannot map are dropped. E3 constraints require both `seq_e3_boltz` and `seq_e3_uniprot`;
    without them they are dropped.
    '''
    uniprot2boltz_poi={uni:boltz for boltz,uni in idxmap_boltz2uniprot(seq_poi_boltz,seq_poi_uniprot).items() if uni>0}
    uniprot2boltz_e3={}
    if seq_e3_boltz and seq_e3_uniprot:
        uniprot2boltz_e3={uni:boltz for boltz,uni in idxmap_boltz2uniprot(seq_e3_boltz,seq_e3_uniprot).items() if uni>0}
    elif pocket_e3_uniprot or covalent_resid_e3_uniprot or contact_ppi_uniprot:
        logging.warning('E3 constraints are given but seq_e3_boltz / seq_e3_uniprot is missing; all E3 constraints are dropped.')
    pocket_poi=[boltz for boltz in (uniprot2boltz_poi.get(int(resid),0) for resid in pocket_poi_uniprot) if boltz]
    pocket_e3=[]
    if uniprot2boltz_e3:
        pocket_e3=[boltz for boltz in (uniprot2boltz_e3.get(int(resid),0) for resid in pocket_e3_uniprot) if boltz]
    contact_ppi=[]
    for poi_resid,e3_resid in contact_ppi_uniprot:
        boltz_poi=uniprot2boltz_poi.get(int(poi_resid),0)
        boltz_e3=uniprot2boltz_e3.get(int(e3_resid),0) if uniprot2boltz_e3 else 0
        if boltz_poi and boltz_e3:
            contact_ppi.append((boltz_poi,boltz_e3))
    covalent_resid_can=uniprot2boltz_poi.get(int(covalent_resid_poi_uniprot),0) if covalent_resid_poi_uniprot else 0
    covalent_resid_e3_can=uniprot2boltz_e3.get(int(covalent_resid_e3_uniprot),0) if (covalent_resid_e3_uniprot and uniprot2boltz_e3) else 0
    return ExperimentConstraints(
        pocket_poi=pocket_poi,
        pocket_e3=pocket_e3,
        contact_ppi=contact_ppi,
        covalent_resid_can=covalent_resid_can,
        covalent_resid_e3_can=covalent_resid_e3_can,
    )


def idxmaps(seqs_boltz_and_uniprot_pair:Iterable[tuple[str,str]]):
    '''Align seq_boltz and seq_uniprot pairs and return a dataframe['seq_boltz','seq_uniprot','idxmap'].'''
    stored_file=Path(__file__).parent/'data/ub_site/idx_boltz2uniprot.csv'
    if os.path.exists(stored_file):
        df=pd.read_csv(stored_file,header='infer')
        df['idxmap']=df['idxmap'].apply(literal_eval)
    else:
        df=pd.DataFrame(columns=['seq_boltz','seq_uniprot','idxmap'])
        df.astype({'seq_boltz':str,'seq_uniprot':str,'idxmap':str})
    current_pairs=set(zip(df['seq_boltz'], df['seq_uniprot']))
    for seq_boltz,seq_uni in seqs_boltz_and_uniprot_pair:
        if (seq_boltz,seq_uni) in current_pairs:
            continue
        try:
            df.loc[len(df)]=[seq_boltz,seq_uni,idxmap_boltz2uniprot(seq_boltz,seq_uni)]
        except Exception as e:
            logging.warning(f'Sequence alignment between `{seq_boltz[:5]}...` and `{seq_uni[:5]}...` failed because: {e}')
            continue
    df.drop_duplicates(['seq_boltz','seq_uniprot'])
    df.copy().astype({'idxmap':str}).to_csv(stored_file,index=False)
    return df


def map_lys_boltz2uniprot_evidence(lys_site_boltz:int|float,idxmap:dict[int,int],df_resi2evidence:pd.DataFrame,uniprot_id:str='')->tuple[float,float]:
    '''Map a Boltz lys residue to uniprot numbering and look up its ubiquitination evidence score.

    The NaN result is preserved instead of silently dropping the record. When the Boltz residue
    cannot be mapped into the uniprot sequence (e.g. it lies in an artificial N-terminal FLAG tag
    or a gene-edited region), or no evidence record exists for the mapped lysine, a NaN fallback
    is returned and the case is reported via `logging.error`.

    Return (lys_site_uniprot, evidence_score); both are NaN when the residue is unresolvable.
    '''
    if pd.isna(lys_site_boltz):
        logging.error(f'{uniprot_id}: no valid Boltz lys residue ({lys_site_boltz}) to map; cannot find the original residue, possibly due to FLAG tag or other gene editing.')
        return float('nan'), float('nan')
    lys_site_uniprot=idxmap.get(int(lys_site_boltz), float('nan'))
    if pd.isna(lys_site_uniprot):
        logging.error(f'{uniprot_id}: Boltz residue {int(lys_site_boltz)} cannot be mapped to uniprot; cannot find the original residue, possibly due to FLAG tag or other gene editing. Keep NaN as fallback.')
        return float('nan'), float('nan')
    evidence_matched=df_resi2evidence[df_resi2evidence['uniprot_lys_resid']==int(lys_site_uniprot)]['evidence_score']
    if evidence_matched.empty:
        logging.error(f'{uniprot_id}: uniprot residue {int(lys_site_uniprot)} has no ubiquitination evidence record; cannot find the original residue. Keep NaN as fallback.')
        return float(lys_site_uniprot), float('nan')
    return float(lys_site_uniprot), float(evidence_matched.iloc[0])


def update_uniprot_uid2seq(uniprot_ids:list[str]):
    '''Update {uniprot id: seq} map into `data/ub_site/id2seq.csv`.

    Return a dataframe['UniprotID','seq'].
    
    `uniprot_ids` don't need to be in the csv file.
    '''
    stored_file=Path(__file__).parent/'data/ub_site/id2seq.csv'
    if os.path.exists(stored_file):
        df=pd.read_csv(stored_file,header='infer')
    else:
        df=pd.DataFrame(columns=['UniprotID','seq'])
        df.astype({'UniprotID':str,'seq':str})
    for uniprot_id in uniprot_ids:
        if uniprot_id not in set(df['UniprotID']):
            try:
                with urllib.request.urlopen(f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json?fields=sequence") as response:
                    data_json = response.read().decode('utf-8')
                    data_dict_uni = json.loads(data_json)
                    uniprot_seq=data_dict_uni['sequence']['value']
            except:
                uniprot_seq=''
            df.loc[len(df)]=[uniprot_id,uniprot_seq]
    df.drop_duplicates('UniprotID').to_csv(stored_file,index=False)
    return df

def uniprot_sequence(uniprot_id:str) -> str:
    '''Return the uniprot sequence of `uniprot_id`, or an empty string when it cannot be fetched.'''
    df_uid2seq=update_uniprot_uid2seq([uniprot_id])
    matched=df_uid2seq[df_uid2seq['UniprotID']==uniprot_id]['seq']
    return matched.iloc[0] if not matched.empty else ''

def e3_uniprot_sequence(name_e3:str) -> str:
    '''Return the uniprot sequence of the E3 ligase `name_e3`.

    Only the E3 ligases in `E3_UNIPROT_ID` are resolvable; an unknown name returns an empty string,
    which makes `cif_boltz2uniprot` warn and leave chain E in boltz numbering.'''
    uniprot_id=E3_UNIPROT_ID.get(name_e3.lower(),'')
    if not uniprot_id:
        logging.warning(f'No uniprot accession for E3 ligase `{name_e3}`; known names are {sorted(E3_UNIPROT_ID)}. Its chain cannot be calibrated to uniprot.')
        return ''
    return uniprot_sequence(uniprot_id)

def fetch_uniprot_ptm_info(uniprot_id:str) -> dict[int,int]:
    '''Fetch and parse ubiquitinylation info from Uniprot and PTMeXchange.

    All Lysines are scored.
    
    The evidence score is 4/3/2 for `Gold/Silver/Bonze`;

    other PTM types like SUMO leads to 1;

    if there is no any evidence, the score is 0.
    '''
    confidence2score={'Gold':EVD_GOLD,'Silver':EVD_SILVER,'Bronze':EVD_BRONZE}
    fields_param = "%2C".join(["sequence","ft_mod_res", "ft_crosslnk", "ft_carbohyd","ft_lipid"]) 
    uni_idx2score=Idx2MaxScore()
    try:
        with urllib.request.urlopen(f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json?fields={fields_param}") as response:
            data_json = response.read().decode('utf-8')
            data_dict_uni = json.loads(data_json)
        uniprot_seq=data_dict_uni['sequence']['value']
        uniprot_lys_indices=set(idx for idx,aa in enumerate(uniprot_seq,1) if aa == 'K')
        uni_idx2score=Idx2MaxScore({lys_idx:0 for lys_idx in uniprot_lys_indices})
        for ptm_feature in data_dict_uni['features']:
            loc=ptm_feature['location']
            loc_start_idx=loc['start']['value']
            if loc_start_idx!=loc['end']['value']:
                continue
            if loc_start_idx not in uniprot_lys_indices:
                continue
            if 'ubiquitin' in ptm_feature['description']:
                uni_idx2score[loc_start_idx]=4
            else:
                uni_idx2score[loc_start_idx]=1
        with urllib.request.urlopen(f'https://www.ebi.ac.uk/proteins/api/proteomics/ptm/{uniprot_id}?format=json') as response:
            data_json = response.read().decode('utf-8')
            data_dict_ex = json.loads(data_json)
        for ptm_feature in data_dict_ex['features']:
            loc_start_idx=int(ptm_feature['begin'])
            for ptm in ptm_feature['ptms']:
                loc_idx=loc_start_idx+ptm['position']-1
                if loc_idx not in uniprot_lys_indices:
                    continue
                if ptm['name']=='Ubiquitinylation':
                    for evidence in ptm['dbReferences']:
                        uni_idx2score[loc_idx]=confidence2score[evidence['properties']['Confidence score']]
                else:
                    uni_idx2score[loc_idx]=1
    except Exception as e:
        logging.error(f'{uniprot_id} failed to get uniprot info because of {e}')
    logging.debug(uni_idx2score.data)
    return uni_idx2score.data

def update_uniprot_ptm_info(uniprot_id:str):
    '''Update and return dataframe['UniprotID', 'uniprotlys_resid', 'evidence_score'] for one uid.'''
    logging.info(f'Fetching PTM info of {uniprot_id}')
    data_path = Path(__file__).parent / 'data/ub_site'
    os.makedirs(data_path, exist_ok=True)
    current_date = datetime.now().date()
    new_uniprot_filename = f"{uniprot_id}_{current_date.strftime('%Y%m%d')}.csv"
    
    exist_uni_files = []
    for file in os.listdir(data_path):
        if file.startswith(f"{uniprot_id}_") and file.endswith('.csv'):
            exist_uni_files.append(file)
    if exist_uni_files:
        exist_uni_files.sort()
        last_date_str=exist_uni_files[-1][:-4].split('_')[-1]
        file_date=datetime.strptime(last_date_str, '%Y%m%d').date()
        if (current_date - file_date).days <= 7:
            [os.remove(data_path/file) for file in exist_uni_files[:-1]]
            try:
                return pd.read_csv(data_path/exist_uni_files[-1],header='infer')
            except:
                pass
    uni_ptm_info=fetch_uniprot_ptm_info(uniprot_id)
    if not uni_ptm_info:
        uni_ptm_info={-10000:-10000}
    df_uni_info=pd.DataFrame(columns=['UniprotID','uniprot_lys_resid','evidence_score'])
    df_uni_info.astype({'UniprotID':str,'uniprot_lys_resid':int,'evidence_score':int})
    for idx, score in uni_ptm_info.items():
        df_uni_info.loc[len(df_uni_info)] = [uniprot_id, idx, score]
    df_uni_info.sort_values(['UniprotID','uniprot_lys_resid'])
    df_uni_info.to_csv(data_path/new_uniprot_filename, index=False)
    return df_uni_info

def update_all_uniprot_ptm_info(uniprot_ids: list[str], max_workers: int = 10):
    '''Update multiple uniprot entries and return a full dataframe['UniprotID', 'uniprotlys_resid', 'evidence_score'] including local cached entries.
    
    Don't set `max_workers` big than 100.'''
    results = []
    min_interval = 0.01
    last_request = 0
    lock = __import__('threading').Lock()
    def fetch_one(uid):
        nonlocal last_request
        with lock:
            now = time.time()
            wait = min_interval - (now - last_request)
            if wait > 0:
                time.sleep(wait)
            last_request = time.time()
        return uid, update_uniprot_ptm_info(uid)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_one, uid): uid for uid in uniprot_ids}
        for future in as_completed(futures):
            uid = futures[future]
            try:
                _, df_uni_info = future.result()
                if df_uni_info is not None and not df_uni_info.empty:
                    results.append(df_uni_info)
            except Exception as e:
                logging.error(e) 
    if not results:
        return pd.DataFrame(columns=['UniprotID','uniprot_lys_resid', 'evidence_score'])
    dfs=pd.concat(results, axis=0, ignore_index=True)
    dfs.to_csv('data/ub_site/uniprot_evidences.csv',index=False)
    return dfs
    
def remap_chain_boltz2uniprot(chain:str,seq_boltz:str,seq_uniprot:str) -> int:
    '''Renumber one chain of the loaded structure from boltz numbering to uniprot numbering.

    Return the number of boltz residues that the alignment leaves unmatched; those residues keep
    `boltz resid + UNMAPPED_RESI_OFFSET` so they never collide with a genuine uniprot index
    (unmatched residues come from artificial tags or gene editing).
    '''
    df_idxmap=idxmaps([(seq_boltz,seq_uniprot)])
    matched=df_idxmap[(df_idxmap['seq_boltz']==seq_boltz)&(df_idxmap['seq_uniprot']==seq_uniprot)]
    idxmap:dict[int,int]=matched.iloc[0]['idxmap'] if not matched.empty else {}
    cmd.alter(f'chain {chain}',f'resi=int(resi)+{UNMAPPED_RESI_OFFSET}')
    unmapped_resid=0
    for idx_boltz in range(1,len(seq_boltz)+1):
        idx_uniprot=idxmap.get(idx_boltz,-1)
        if idx_uniprot<=0:
            unmapped_resid+=1
            idx_uniprot=idx_boltz+UNMAPPED_RESI_OFFSET
        cmd.alter(f'chain {chain} and resi {idx_boltz+UNMAPPED_RESI_OFFSET}',f'resi={idx_uniprot}')
    return unmapped_resid

def cif_boltz2uniprot(cif_file:str,seq_poi_boltz:str,seq_poi_uniprot:str,seq_e3_boltz:str='',seq_e3_uniprot:str=''):
    '''Reindex resid for boltz prediction structures to be according with uniprot.

    The POI (chain P) is always remapped. The E3 ligase (chain E) is remapped only when both
    `seq_e3_boltz` and `seq_e3_uniprot` are given; without that pair chain E keeps the boltz
    numbering and the skipped calibration is reported through `logging.warning`.

    Conversion rate is about 27aa/s per chain.'''
    cif_file_path=Path(os.path.abspath(cif_file))
    cif_file_uniprot_path=cif_file_path.parent/(cif_file_path.stem+'_uni.cif')
    cmd.reinitialize()
    cmd.load(str(cif_file_path))
    unmapped_poi=remap_chain_boltz2uniprot('P',seq_poi_boltz,seq_poi_uniprot)
    if unmapped_poi:
        logging.warning(f'{cif_file_path.name}: {unmapped_poi} POI residues cannot be aligned to uniprot and keep resid + {UNMAPPED_RESI_OFFSET}, mainly because of artificial gene edition or mutation.')
    if seq_e3_boltz and seq_e3_uniprot:
        if cmd.count_atoms('chain E')>0:
            unmapped_e3=remap_chain_boltz2uniprot('E',seq_e3_boltz,seq_e3_uniprot)
            if unmapped_e3:
                logging.warning(f'{cif_file_path.name}: {unmapped_e3} E3 residues cannot be aligned to uniprot and keep resid + {UNMAPPED_RESI_OFFSET}, mainly because of artificial gene edition or mutation.')
        else:
            logging.warning(f'{cif_file_path.name}: no chain E in the structure; the E3 ligase is not calibrated to uniprot.')
    elif seq_e3_boltz or seq_e3_uniprot:
        logging.warning(f'{cif_file_path.name}: both seq_e3_boltz and seq_e3_uniprot are required to calibrate the E3 ligase; chain E keeps the boltz numbering.')
    else:
        logging.warning(f'{cif_file_path.name}: no E3 ligase sequence given; chain E keeps the boltz numbering and is NOT calibrated to uniprot.')
    cmd.save(cif_file_uniprot_path)

def read_boltz_chain_sequence(boltz_yaml:str,chain_id:str) -> str:
    '''Read the protein sequence of `chain_id` from a boltz input yaml.

    Return an empty string when the yaml holds no protein entry for that chain, so the caller can pass
    the result straight to `cif_boltz2uniprot` and receive its warning about a skipped calibration.'''
    with open(boltz_yaml,encoding='utf-8') as yaml_file:
        boltz_config=yaml.safe_load(yaml_file) or {}
    for entry in boltz_config.get('sequences',[]):
        protein=entry.get('protein')
        if protein and chain_id in protein.get('id',[]):
            return protein.get('sequence','')
    return ''

def write_boltz_template_cif(
    cif_file:str,out_cif:str='',
    required_chains:tuple[str,...]=('P','E'),
) -> str:
    '''Write a boltz template CIF from a boltz prediction `*_model_0.cif`.

    Return the path of the written template.'''
    cif_path=Path(os.path.abspath(cif_file))
    out_path=Path(out_cif) if out_cif else cif_path.parent/(cif_path.stem+'_template.cif')
    structure=gemmi.read_structure(str(cif_path))
    structure.setup_entities()
    for chain_name in [chain.name for chain in structure[0] if chain.name not in required_chains]:
        structure[0].remove_chain(chain_name)
    structure.remove_ligands_and_waters()
    structure.remove_empty_chains()
    structure.remove_hydrogens()
    structure.remove_alternative_conformations()
    polymer_chains=sorted(chain.name for chain in structure[0] if len(chain.get_polymer())>0)
    missing=[name for name in required_chains if name not in polymer_chains]
    if missing:
        raise ValueError(f'{cif_path.name}: template needs chains {list(required_chains)} but the structure has {polymer_chains}; missing {missing}. Rename them in the source CIF or use the boltz prediction `*_model_0.cif`.')
    document=structure.make_mmcif_document()
    document.write_file(str(out_path))
    block=gemmi.cif.read(str(out_path))[0]
    entity_ids=block.find_values('_entity_poly.entity_id')
    sequences=block.find_values('_entity_poly.pdbx_seq_one_letter_code_can') or block.find_values('_entity_poly.pdbx_seq_one_letter_code')
    if not sequences:
        raise ValueError(f'{out_path.name}: the template has no _entity_poly sequence, boltz cannot expand its residues. Rebuild it from a CIF that keeps _entity_poly, e.g. the prediction `*_model_0.cif`.')
    if len(entity_ids)!=len(polymer_chains) or any(len(sequence)<2 for sequence in sequences):
        raise ValueError(f'{out_path.name}: _entity_poly holds {len(entity_ids)} entries with lengths {[len(sequence) for sequence in sequences]} for {len(polymer_chains)} chains; the source CIF has no usable _entity_poly, use the boltz prediction `*_model_0.cif` instead.')
    return str(out_path)


def read_covalent_bonds(ternary_yaml:str,ligand_chain:str='L') -> list[dict[str,tuple[str,int,str]]]:
    '''Read every covalent bond constraint of a boltz input yaml.

    Each item holds keys `ligand` and `protein`, values are (chain, resid, atom name). A PROTAC with
    warheads on both ends yields one item whose protein chain is `P` and one whose protein chain is
    `E`. Residue numbering follows the boltz chain, so the protein resid still needs
    `cif_boltz2uniprot` mapping before it matches a `*_uni.cif`; the ligand side is unaffected.
    An empty list means the yaml holds no bond constraint (a non-covalent run).'''
    bonds:list[dict[str,tuple[str,int,str]]]=[]
    sides:list[tuple[str,int,str]]=[]
    with open(ternary_yaml,encoding='utf-8') as yaml_file:
        for line in yaml_file:
            if not line.strip().startswith(('atom1:','atom2:')):
                continue
            chain,resid,atom_name=[item.strip() for item in line.split('[',1)[1].split(']',1)[0].split(',')]
            sides.append((chain,int(resid),atom_name))
    ligand_sides=[side for side in sides if side[0]==ligand_chain]
    protein_sides=[side for side in sides if side[0]!=ligand_chain]
    for ligand_side,protein_side in zip(ligand_sides,protein_sides):
        bonds.append({'ligand':ligand_side,'protein':protein_side})
    return bonds

def read_covalent_bond(ternary_yaml:str,ligand_chain:str='L') -> dict[str,tuple[str,int,str]]:
    '''Read the first covalent bond constraint of a boltz input yaml.

    Kept for callers that handle a single warhead; use `read_covalent_bonds` when the E3 side may
    carry its own warhead. An empty dict means the yaml holds no bond constraint.'''
    bonds=read_covalent_bonds(ternary_yaml,ligand_chain)
    return bonds[0] if bonds else {}

# the covalent ligand atom name follows the atom order of each molecule, so pass covalent_yaml (the boltz input yaml of the same molecule) to read the bond constraint
def generate_pymol_pse(cif_file:str,lys_site:int,ref_cif_poi:str='',ref_cif_align_selection:str='',ref_cif_e3:str='',ref_cif_ternary:str='',covalent_resid:int=0,covalent_res_atomname='SG',covalent_lig_atomname:str='',covalent_yaml:str='',covalent_e3_resid:int=0,covalent_e3_res_atomname='SG',covalent_e3_lig_atomname:str='',):
    '''Create a pymol project session file, including the predicted machinery+poi structure, Ub/lys distance, clash atoms, reference structures and covalent bond.

    `cif_file` is expected to be the `*_uni.cif` written by `cif_boltz2uniprot`, so chain P and, when
    E3 sequences were passed there, chain E already carry uniprot residue numbering.

    The covalent ligand atom name depends on the atom order of each molecule, so pass covalent_yaml
    (the boltz input yaml of the same molecule) to read the bond constraints instead of hard-coding
    them. A warhead on the E3 side is drawn when `covalent_e3_resid` is set; pass `covalent_yaml` to
    pick up its atom names, otherwise the E3 warhead is skipped.

    Press `F1` to see panorama view and `F2` to see closeup view.'''
    try:
        lys_site=int(lys_site)  # pandas may read it as a float (e.g. 185.0)
        if covalent_yaml:
            # the ligand atom name follows the atom order of each molecule, so it is read from the boltz input
            covalent_bonds=read_covalent_bonds(covalent_yaml)
            if not covalent_bonds:
                logging.warning(f'No covalent bond constraint in {covalent_yaml}, the covalent bond is skipped.')
                covalent_resid=0
                covalent_e3_resid=0
            else:
                poi_bonds=[bond for bond in covalent_bonds if bond['protein'][0]=='P']
                e3_bonds=[bond for bond in covalent_bonds if bond['protein'][0]=='E']
                if poi_bonds:
                    covalent_lig_atomname=poi_bonds[0]['ligand'][2]
                    covalent_res_atomname=poi_bonds[0]['protein'][2]
                if e3_bonds:
                    covalent_e3_lig_atomname=e3_bonds[0]['ligand'][2]
                    covalent_e3_res_atomname=e3_bonds[0]['protein'][2]
        if covalent_resid:
            covalent_resid=int(covalent_resid)
        if covalent_e3_resid:
            covalent_e3_resid=int(covalent_e3_resid)
        # panorama view
        cmd.reinitialize()
        cmd.set('label_size',30)
        cmd.set('label_font_id',7)
        cmd.load(cif_file,'panorama')
        cmd.space('rgb')
        cmd.bg_color('white')
        cmd.color('yelloworange')
        cmd.color('pink','chain E')
        cmd.color('slate','chain P')
        cmd.color('cyan','chain D') # E2
        cmd.color('green','chain U')
        cmd.color('magenta','chain L')
        cmd.util.cnc('chain L')
        cmd.scene('F1','store','panorama view')
        cmd.disable('panorama')

        # close view
        cmd.copy_to('close','panorama')
        cmd.group('close_group','close','add')
        ## reference structures
        if ref_cif_poi:
            cmd.load(ref_cif_poi,'ref_poi')
            ref_cif_align_selection='ref_poi'+(f' and {ref_cif_align_selection}' if ref_cif_align_selection else "")
            cmd.super(ref_cif_align_selection,'panorama and chain P')
            cmd.group('close_group','ref_poi','add')
        if ref_cif_e3:
            cmd.load(ref_cif_e3,'ref_e3')
            cmd.super('ref_e3','panorama and chain E')
            cmd.group('close_group','ref_e3','add')
        if ref_cif_ternary:
            cmd.load(ref_cif_ternary,'ref_ternary')
            cmd.super('ref_ternary','panorama and chain E+P')
            cmd.group('close_group','ref_ternary','add')
        cmd.color('white','ref_*')
        cmd.set('stick_transparency',0.2,'ref_*')
        cmd.remove('ref_* and polymer')
        cmd.set('cartoon_transparency',0.6,'close')
        ## covalent bond
        if covalent_resid and covalent_lig_atomname:
            sele_covalent_res=f'close and chain P and resi {covalent_resid}'
            sele_covalent_lig=f'close and chain L and name {covalent_lig_atomname}'
            if cmd.count_atoms(sele_covalent_res)>0 and cmd.count_atoms(sele_covalent_lig)>0:
                cmd.show('sticks',sele_covalent_res)
                cmd.bond(f'{sele_covalent_res} and name {covalent_res_atomname}',sele_covalent_lig)
                cmd.copy_to('cov_poi',sele_covalent_res)
                cmd.group('close_group','cov_poi','add')
                cmd.label(f'cov_poi and name {covalent_res_atomname}','oneletter+resi')
        elif covalent_resid:
            logging.warning('covalent_resid is set but no ligand atom name is available; pass covalent_yaml or covalent_lig_atomname to draw the covalent bond.')
        ## E3 covalent bond
        if covalent_e3_resid and covalent_e3_lig_atomname:
            sele_covalent_e3_res=f'close and chain E and resi {int(covalent_e3_resid)}'
            sele_covalent_e3_lig=f'close and chain L and name {covalent_e3_lig_atomname}'
            if cmd.count_atoms(sele_covalent_e3_res)>0 and cmd.count_atoms(sele_covalent_e3_lig)>0:
                cmd.show('sticks',sele_covalent_e3_res)
                cmd.bond(f'{sele_covalent_e3_res} and name {covalent_e3_res_atomname}',sele_covalent_e3_lig)
                cmd.copy_to('cov_e3',sele_covalent_e3_res)
                cmd.group('close_group','cov_e3','add')
                cmd.label(f'cov_e3 and name {covalent_e3_res_atomname}','oneletter+resi')
        elif covalent_e3_resid:
            logging.warning('covalent_e3_resid is set but no E3 ligand atom name is available; pass covalent_yaml or covalent_e3_lig_atomname to draw the E3 covalent bond.')
        ## Ub/Lys distance (skip when the lys residue or Ub G75 is missing)
        sele_lys_poi=f'close and chain P and resi {lys_site}'
        sele_gly_ub='close and chain U and resi 75'
        if cmd.count_atoms(sele_lys_poi)>0 and cmd.count_atoms(sele_gly_ub)>0:
            cmd.show('sticks',sele_lys_poi)
            cmd.show('sticks',sele_gly_ub)
            cmd.distance('dist_ublys',f'{sele_lys_poi} and name NZ',f'{sele_gly_ub} and name C')
            cmd.group('close_group','dist_ublys','add')
            cmd.copy_to('lys_poi', sele_lys_poi)
            cmd.label('lys_poi and name NZ', 'oneletter+resi')
            cmd.group('close_group','lys_poi','add')
        ## clash atoms (skip if there is no clash within 1 A)
        sele_clash='close and (chain P near_to 1 of (not chain P+E+L))'
        if cmd.count_atoms(sele_clash)>0:
            cmd.copy_to('clash', sele_clash)
            cmd.group('close_group','clash','add')
            cmd.show('sphere','clash')
            cmd.set('sphere_color','red')
            cmd.set('sphere_scale',0.2)
            cmd.hide('cartoon','clash')
        cmd.enable('not panorama')
        cmd.zoom('close and chain E+L+P+U')
        cmd.util.cnc()
        cmd.scene('F2','store','close view')
        cmd.save(Path(cif_file).parent/(Path(cif_file).stem+'.pse'))
    except Exception:
        logging.exception('generate_pymol_pse failed')