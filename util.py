from ast import literal_eval
from pymol import cmd
from collections import UserDict
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
import shutil
from typing import Iterable
import urllib.request
import json
import time
from boltz.data.msa.mmseqs2 import run_mmseqs2
import pandas as pd
from rdkit.Chem import MolFromSmiles,MolToInchiKey
from DockQ.DockQ import load_PDB, run_on_all_native_interfaces
from concurrent.futures import ThreadPoolExecutor, as_completed
from Bio.Align import PairwiseAligner

EVD_GOLD=4
EVD_SILVER=3
EVD_BRONZE=2
EVD_OTHER=1
EVD_NONE=0

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

def idxmaps(seqs_boltz_and_uniprot_pair:Iterable[tuple[str,str]]):
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
            logging.warning(f'Sequence alignment between {seq_boltz[:5]} and {seq_uni[:5]} faild because: {e}')
            continue
    df.drop_duplicates(['seq_boltz','seq_uniprot'])
    df.copy().astype({'idxmap':str}).to_csv(stored_file,index=False)
    return df
   
def fetch_uniprot_uid2seq(uniprot_ids:list[str]):
    stored_file=Path(__file__).parent/'data/ub_site/id2seq.csv'
    if os.path.exists(stored_file):
        df=pd.read_csv(stored_file,header='infer')
    else:
        df=pd.DataFrame(columns=['UniprotID','seq'])
        df.astype({'UniprotID':str,'seq':str})
    for uniprot_id in uniprot_ids:
        if df['UniprotID'].isin([uniprot_id]).any():
            try:
                with urllib.request.urlopen(f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json?fields=sequence") as response:
                    data_json = response.read().decode('utf-8')
                    data_dict_uni = json.loads(data_json)
                    uniprot_seq=data_dict_uni['sequence']['value']
            except:
                uniprot_seq=''
            df.loc[len(df)]=[uniprot_id,uniprot_seq]
    df.to_csv(stored_file,index=False)
    return df

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

def update_uniprot_ptm_info(uniprot_id):
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

def update_all_uniprot_ub_info(uniprot_ids: list[str], max_workers: int = 10):
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
    
def cif_boltz2uniprot(cif_file,seq_poi_boltz,seq_poi_uniprot):
    cif_file_path=Path(os.path.abspath(cif_file))
    cif_file_uniprot_path=cif_file_path.parent/(cif_file_path.stem+'_uni.cif')
    df_idxmap=idxmaps([(seq_poi_boltz,seq_poi_uniprot)])
    idxmap:dict=df_idxmap[(df_idxmap['seq_boltz']==seq_poi_boltz) &(df_idxmap['seq_uniprot']==seq_poi_uniprot)].iloc[0]['idxmap']
    cmd.reinitialize()
    cmd.load(str(cif_file_path))
    cmd.alter(f'chain P',f'resi=int(resi)+5000')
    for idx_boltz_python in range(len(seq_poi_boltz)):
        idx_boltz=idx_boltz_python+1
        idx_uniprot=idxmap.get(idx_boltz,idx_boltz+5000)
        # add 5000 if the lys is not in the idxmap, mainly because of artificial gene edition or mutation
        cmd.alter(f'chain P and resi {idx_boltz+5000}',f'resi={idx_uniprot}')
        cmd.save(cif_file_uniprot_path)