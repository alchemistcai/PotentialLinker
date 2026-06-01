from functools import partial
import logging
from pathlib import Path
import os,sys
from ast import literal_eval
from detect_pocket import get_ppi_info_from_pdb_info
sys.path.append('../../')
os.environ.setdefault('BOLTZ_CACHE', '~/projects/boltz')
# os.environ['BOLTZ_CACHE']= '/root/autodl-tmp/boltz' # change boltz cache path
import pandas as pd
from util import idxmap_boltz2uniprot, smi2inchikey, update_uniprot_ptm_info, update_uniprot_uid2seq,get_a3m,dockq_for_protac
from reinvent_plugins.components.comp_boltz_fgfr4 import calculate_one_affinity_score
from pymol import cmd
import warnings

warnings.filterwarnings('ignore',category=UserWarning)
logging.basicConfig(level=logging.INFO)

def make_score_df(records,restraint_type:str):
    sc_df=pd.DataFrame(records)
    sc_df.columns=[f'{restraint_type}_{col}' for col in sc_df.columns]
    return sc_df

def annotate_score_dict(result_calculate_one_affinity_score:tuple[float,dict],idxmap) -> dict:
    try:
        lys_site_uniprot=idxmap[result_calculate_one_affinity_score[1]['closest_lys_resid']]
        evidence_score=df_resi2evidence[df_resi2evidence['uniprot_lys_resid']==lys_site_uniprot]['evidence_score'].iloc[0]
        return result_calculate_one_affinity_score[1]|{'score':result_calculate_one_affinity_score[0], 'lys_site_uniprot':lys_site_uniprot,'evidence_score':evidence_score}
    except:
        return result_calculate_one_affinity_score[1]|{'score':result_calculate_one_affinity_score[0], 'lys_site_uniprot':-10000,'evidence_score':-10000}


# get PDB files
df=pd.read_csv('pdb14.csv',header='infer',dtype=str)
for pdbid,ligid,chain_poi,chain_lig,chain_e3 in df[['pdbid','ligid','chain_poi','chain_lig','chain_e3']].itertuples(False):
    try:
        pdbid=pdbid.lower()
        cmd.reinitialize()
        cmd.fetch(pdbid,name=f'{pdbid}',type='cif',path='./')
    except Exception as e:
        logging.warning(e)
        continue
df[['seq_poi','seq_e3','seq_auth2can_poi','seq_auth2can_e3','smi','ppi','pocket_poi','pocket_e3','atomname_ccd2boltz']]=df.apply(lambda row:get_ppi_info_from_pdb_info(row['pdbid'],row['ligid'],row['chain_poi'],row['chain_lig'],row['chain_e3'],workdir='./'),axis=1,result_type='expand')
df['inchikey']=df['smi'].apply(smi2inchikey)
df=df.astype(str)
df.to_csv('pdb14_extend.csv',index=False)

# fetch MSA
import time
df_extend=pd.read_csv('pdb14_extend.csv',header='infer',dtype=str)
for seq_poi,seq_e3,pdbid in df_extend[['seq_poi','seq_e3','pdbid']].itertuples(False):
    time.sleep(0.3)
    pdbid=pdbid.lower()
    get_a3m(seq_poi,f'{pdbid}_poi')
    get_a3m(seq_e3,f'{pdbid}_e3')

# calculate NWHM scores and annotate lys evidence scores
df_extend=pd.read_csv('pdb14_extend.csv',header='infer',dtype=str)
df_extend['inchikey']=df_extend['smi'].apply(smi2inchikey)
results_ppi_tmp=[]
results_ppi=[]
results_tmp=[]

for lig_smi,inchikey,pdbid,seq_poi,pocket_poi,seq_e3,pocket_e3,ppi,chain_poi_can,chain_e3_can,name_e3_origin,uniprot_id in df_extend[['smi','inchikey','pdbid','seq_poi','pocket_poi','seq_e3','pocket_e3','ppi','chain_poi_can','chain_e3_can','E3 Ligase','UniprotID']].itertuples(False):
    try:
        df_resi2evidence=update_uniprot_ptm_info(uniprot_id)
        df_uid2seq=update_uniprot_uid2seq([uniprot_id])
        df_uid2seq=df_uid2seq[df_uid2seq['UniprotID']==uniprot_id]
        seq_uniprot=df_uid2seq['seq'].iloc[0]
        idxmap=idxmap_boltz2uniprot(seq_poi,seq_uniprot)
        pdbid=pdbid.lower()
        name_poi=pdbid+'_poi'
        name_e3:str=pdbid+'_e3'
        ub_ref_rootdir=Path(__file__).parent.parent.parent/'data/crl_models'
        if name_e3_origin in ('vhl','crbn'):
            if name_e3_origin=='vhl':
                ub_ref_path=ub_ref_rootdir/'vhl/'
            elif name_e3_origin=='crbn':
                ub_ref_path=ub_ref_rootdir/'crbn/'
            try:
                (ub_ref_rootdir/name_e3).symlink_to(ub_ref_path,True)
            except:
                pass
        pocket_poi=literal_eval(pocket_poi)
        pocket_e3=literal_eval(pocket_e3)
        ppi=literal_eval(ppi)
        template_cif_or_path=f'{pdbid}.cif'
        calc_given_score=partial(calculate_one_affinity_score,lig_smi=lig_smi,name_poi=name_poi,seq_poi=seq_poi,pocket_poi=pocket_poi,name_e3=name_e3,seq_e3=seq_e3,pocket_e3=pocket_e3,msa_path='./',name_e3_origin=name_e3_origin,
        # extract_from_precomputed=False, # calculate it by yourself
        extract_from_precomputed=True
        )

        result_ppi_tmp=calc_given_score(output_path='ppi_tmp',
        pocket_poi=pocket_poi,pocket_e3=pocket_e3,
        contact_ppi=ppi,
        template_cif_or_path=template_cif_or_path,template_chain_poi_can=chain_poi_can,template_chain_e3_can=chain_e3_can,)
        results_ppi_tmp.append(annotate_score_dict(result_ppi_tmp,idxmap))

        result_tmp=calc_given_score(output_path='tmp',pocket_poi=pocket_poi,pocket_e3=pocket_e3,
        contact_ppi=None,
        template_cif_or_path=template_cif_or_path,template_chain_poi_can=chain_poi_can,template_chain_e3_can=chain_e3_can,)
        results_tmp.append(annotate_score_dict(result_tmp,idxmap))

        result_ppi=calc_given_score(output_path='ppi',
        pocket_poi=pocket_poi,pocket_e3=pocket_e3,
        contact_ppi=ppi,
        template_cif_or_path='',)
        results_ppi.append(annotate_score_dict(result_ppi,idxmap))
    except Exception as e:
        logging.error(e,exc_info=True)
        continue
logging.debug(pd.DataFrame(results_ppi_tmp))
sc_df_ppi_tmp=make_score_df(results_ppi_tmp,'ppi_tmp')
sc_df_ppi=make_score_df(results_ppi,'ppi')
sc_df_tmp=make_score_df(results_tmp,'tmp')
logging.debug(sc_df_ppi_tmp)
sc_df=pd.concat((df_extend,sc_df_ppi_tmp,sc_df_ppi,sc_df_tmp),axis=1)
logging.info(sc_df.head())
sc_df.to_csv('pdb14_precompute.csv',index=False)

# calculate DockQ
df_extend['pdbid']=df_extend['pdbid'].str.lower()
def calc_dockq_ablation(root_dir:str,df:pd.DataFrame):
    df[[f'{root_dir}_PE_DockQ',f'{root_dir}_PE_iRMSD',f'{root_dir}_PE_LRMSD',f'{root_dir}_PE_fnat',f'{root_dir}_PL_LRMSD',f'{root_dir}_EL_LRMSD']]=df.apply(lambda row:dockq_for_protac(f'clean/{row["pdbid"]}_ref.cif',f'{root_dir}/boltz_results_{row["pdbid"]}_poi_{row["pdbid"]}_e3_{row["inchikey"]}_ternary/predictions/{row["pdbid"]}_poi_{row["pdbid"]}_e3_{row["inchikey"]}_ternary/{row["pdbid"]}_poi_{row["pdbid"]}_e3_{row["inchikey"]}_ternary_model_0.cif'),axis=1,result_type='expand')
predict_dirs=['ppi','tmp','ppi_tmp']
for root_dir in predict_dirs:
    logging.info(f'Calculating dockq in {root_dir} ...')
    calc_dockq_ablation(root_dir,df_extend)
df_extend.to_csv('pdb14_dockq.csv',index=False)