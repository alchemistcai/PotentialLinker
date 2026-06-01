import shutil
import os,sys
sys.path.append('../../')
os.environ.setdefault('BOLTZ_CACHE', '~/projects/boltz') # change boltz cache path
from ast import literal_eval
from pymol import cmd
import pandas as pd
from detect_pocket import get_pocket_info_from_pdb_info
from util import update_uniprot_ptm_info,update_uniprot_uid2seq,idxmap_boltz2uniprot
from util import cif_boltz2uniprot,generate_pymol_pse,get_a3m,smi2inchikey
from reinvent_plugins.components.comp_boltz_fgfr4 import calculate_one_affinity_score

# get PDB file
pdbid_stat3='6njs'
cmd.reinitialize()
cmd.fetch(pdbid_stat3,name=pdbid_stat3,type='cif',path='./')
stat3_seq,stat3_pocket=get_pocket_info_from_pdb_info(pdbid_stat3,'KQV','A','A','./',4)

# fetch MSA
get_a3m(stat3_seq,pdbid_stat3)
shutil.copy('../../data/MSA/crbn.a3m','crbn.a3m')

# get uniprot info
df_resi2evidence=update_uniprot_ptm_info('P40763')
df_uid2seq=update_uniprot_uid2seq(['P40763'])
df_uid2seq=df_uid2seq[df_uid2seq['UniprotID']=='P40763']
seq_uniprot=df_uid2seq['seq'].iloc[0]
idxmap=idxmap_boltz2uniprot(stat3_seq,seq_uniprot)

# calculate NWHM scores
pdb_df=pd.read_csv('stat3_crbn_protac.csv',header='infer')
pdb_df['pdbid']=pdb_df['pdbid'].str.lower()
pdb_df['inchikey']=pdb_df['PROTAC SMILES'].apply(smi2inchikey)
tmp_result=[]
for lig_smi,inchikey,name_poi,seq_poi,pocket_poi,name_e3 in pdb_df[['PROTAC SMILES','inchikey','pdbid','seq_can','boltz_pocket','E3 Ligase']].itertuples(False):
    result=calculate_one_affinity_score(lig_smi,name_poi,seq_poi,literal_eval(pocket_poi),name_e3,output_path='./',msa_path='./',extract_from_precomputed=False)
    tmp_result.append(result[1]|{'score':result[0]})
sc_df=pd.DataFrame(tmp_result)
pedia_precompute=pd.concat([pdb_df,sc_df],axis=1)

# annotate uniprot lys sites
pedia_precompute.rename(columns={'closest_lys_resid':'lys_site_boltz'},inplace=True)
pedia_precompute['lys_site_uniprot']=pedia_precompute['lys_site_boltz'].map(idxmap)
pedia_precompute=pedia_precompute.merge(df_resi2evidence,left_on='lys_site_uniprot',right_on='uniprot_lys_resid')
pedia_precompute.to_csv('stat3_precompute.csv',index=False)

# generate pse files
for name_poi,name_e3,inchikey,lys_site_uniprot in pedia_precompute[['pdbid','E3 Ligase','inchikey','lys_site_uniprot']].itertuples(False):
    best_cif=f'{name_poi}_{name_e3}_{inchikey}_ternary_best.cif'
    cif_boltz2uniprot(best_cif,stat3_seq,seq_uniprot)
    generate_pymol_pse(best_cif[:-4]+'_uni.cif',lys_site_uniprot,'clean/6njs_ref.cif','resi 580-670','../../data/PDB/clean/8rqa_ref.cif') # resi 580-670 is SH2 domain of STAT3