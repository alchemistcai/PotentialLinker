import pandas as pd
from reinvent_plugins.components.boltz_interface import calculate_affinity_score,run_superpose_scoring_items_calc
from ast import literal_eval
import os
pedia_extend=pd.read_csv('data/pedia_extend.csv',header='infer')


os.environ['BOLTZ_CACHE']= '/root/autodl-tmp/boltz'
tmp_result=[]
for lig_smi,inchikey,name_poi,seq_poi,pocket_poi,name_e3 in pedia_extend[['PROTAC SMILES','inchikey','pdbid','seq_can','boltz_pocket','E3 Ligase']].itertuples(False):
    tmp_result.extend(calculate_affinity_score(lig_smi,inchikey,name_poi,seq_poi,literal_eval(pocket_poi),name_e3,extract_from_precomputed=True))
sc_df=pd.DataFrame(tmp_result,columns=['pdbid','E3 Ligase','inchikey','lgKd_poi','lgKd_e3','lgKd_ternary','ref_model','dist_ub','has_surf_lys','bsa','num_clash'])
sc_df=sc_df.astype({'lgKd_poi':float,'lgKd_e3':float,'lgKd_ternary':float,'ref_model':str,'dist_ub':float,'has_surf_lys':int,'bsa':float,'num_clash':float})
pedia_precompute=pd.merge(pedia_extend,sc_df,on=['pdbid', 'E3 Ligase', 'inchikey'], how='inner').drop_duplicates(['pdbid', 'E3 Ligase', 'inchikey','ref_model'],keep='first') # activity in different cells are different records
pedia_precompute['Active/Inactive']=(pedia_precompute['Active/Inactive']=='Active').astype(int)
pedia_precompute['lg_alpha']=pedia_precompute['lgKd_poi']+pedia_precompute['lgKd_e3']-pedia_precompute['lgKd_ternary']
pedia_precompute.astype({'PROTACDB ID':int}).to_csv('data/pedia_precompute.csv',index=False)