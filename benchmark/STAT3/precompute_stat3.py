import logging
import re
import shutil
import os,sys
sys.path.append('../../')
os.environ.setdefault('BOLTZ_CACHE', '~/projects/boltz') # change boltz cache path
from ast import literal_eval
from pymol import cmd
import matplotlib
matplotlib.use('Agg') # only the saved figure is needed, no interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from detect_pocket import get_pocket_info_from_pdb_info
from util import update_uniprot_ptm_info,update_uniprot_uid2seq,idxmap_boltz2uniprot
from util import cif_boltz2uniprot,generate_pymol_pse,get_a3m,smi2inchikey,map_lys_boltz2uniprot_evidence
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
pdb_df['seq_can']=stat3_seq
pdb_df['boltz_pocket']=str(stat3_pocket)
tmp_result=[]
for lig_smi,inchikey,name_poi,seq_poi,pocket_poi,name_e3 in pdb_df[['PROTAC SMILES','inchikey','pdbid','seq_can','boltz_pocket','E3 Ligase']].itertuples(False):
    result=calculate_one_affinity_score(lig_smi,name_poi,seq_poi,literal_eval(pocket_poi),name_e3,output_path='./',msa_path='./',extract_from_precomputed=False)
    tmp_result.append(result[1]|{'score':result[0]})
sc_df=pd.DataFrame(tmp_result)
pedia_precompute=pd.concat([pdb_df,sc_df],axis=1)

# annotate uniprot lys sites; NaN results are kept (and reported) instead of being dropped
pedia_precompute.rename(columns={'closest_lys_resid':'lys_site_boltz'},inplace=True)
uniprot_id='P40763'
annotated=pedia_precompute['lys_site_boltz'].apply(lambda lys_boltz: map_lys_boltz2uniprot_evidence(lys_boltz,idxmap,df_resi2evidence,uniprot_id))
pedia_precompute['lys_site_uniprot']=[site for site,_ in annotated]
pedia_precompute['uniprot_lys_resid']=pedia_precompute['lys_site_uniprot']
pedia_precompute['evidence_score']=[score for _,score in annotated]
pedia_precompute['UniprotID']=uniprot_id

# parse the measured DC50; records above the assay upper limit keep the limit value and are dropped from the fit
DC50_PATTERN=r'\s*([<>]?)\s*([0-9]*\.?[0-9]+)'


def parse_DC50(value):
    """Split '1.8' / '>5' into the numeric value and whether it is censored by the assay upper limit."""
    match=re.match(DC50_PATTERN,str(value))
    if match is None:
        return np.nan,True
    return float(match.group(2)),bool(match.group(1))


dc50_parsed=[parse_DC50(value) for value in pedia_precompute['DC50']]
pedia_precompute['DC50_uM']=[value for value,_ in dc50_parsed]
pedia_precompute['DC50_censored']=[censored for _,censored in dc50_parsed]
pedia_precompute['logDC50']=np.log10(pedia_precompute['DC50_uM'])
print(f'parsed DC50 for {int(pedia_precompute["DC50_uM"].notna().sum())} of {len(pedia_precompute)} records; '
      f'{int(pedia_precompute["DC50_censored"].sum())} records are above the assay upper limit, plotted at that '
      f'limit and excluded from R2')

pedia_precompute.to_csv('stat3_precompute.csv',index=False)

# generate pse files; skip records whose lys site cannot be mapped to uniprot (e.g. FLAG/gene-edited)
for name_poi,name_e3,inchikey,lys_site_uniprot in pedia_precompute[['pdbid','E3 Ligase','inchikey','lys_site_uniprot']].itertuples(False):
    if pd.isna(lys_site_uniprot):
        logging.warning(f'Skip pse for {name_poi}_{name_e3}_{inchikey}: lys site cannot be mapped to uniprot (FLAG/gene editing).')
        continue
    best_cif=f'{name_poi}_{name_e3}_{inchikey}_ternary_best.cif'
    cif_boltz2uniprot(best_cif,stat3_seq,seq_uniprot)
    generate_pymol_pse(best_cif[:-4]+'_uni.cif',lys_site_uniprot,'clean/6njs_ref.cif','resi 580-670','../../data/PDB/clean/8rqa_ref.cif') # resi 580-670 is SH2 domain of STAT3

# NWHM vs log10(DC50); filled markers are quantitative records, hollow markers sit on the assay upper limit
fig,ax=plt.subplots(figsize=(5,5))
censored_dc50=pedia_precompute.loc[pedia_precompute['DC50_censored'],'DC50_uM'].dropna()
upper_limit=f'{censored_dc50.max():g} μM' if len(censored_dc50)>0 else 'upper limit'
for censored,quality in [(False,f'DC50 ≤ {upper_limit}'),(True,f'DC50 > {upper_limit}')]:
    subset=pedia_precompute[pedia_precompute['DC50_censored']==censored]
    if subset.empty:
        continue
    ax.scatter(subset['score'],subset['logDC50'],s=45,color='tab:blue',zorder=3,
               facecolors='none' if censored else 'tab:blue',edgecolors='tab:blue',
               label=f'{quality} (n = {len(subset)})')
for level in pedia_precompute.loc[pedia_precompute['DC50_censored'],'logDC50'].dropna().unique():
    ax.axhline(level,color='gray',linestyle='--',linewidth=1.2) # upper limit of the assay

# R2 and its interval use only the records with a quantitative DC50
R2_BOOTSTRAP_RESAMPLES=2000
R2_BOOTSTRAP_SEED=0
measured=pedia_precompute[~pedia_precompute['DC50_censored']].dropna(subset=['score','logDC50'])
measured_score=measured['score'].to_numpy(dtype=float)
measured_log_dc50=measured['logDC50'].to_numpy(dtype=float)
r2=stats.pearsonr(measured_score,measured_log_dc50).statistic**2
slope,intercept=np.polyfit(measured_score,measured_log_dc50,1)

rng=np.random.default_rng(R2_BOOTSTRAP_SEED)
bootstrap_r2=[]
for _ in range(R2_BOOTSTRAP_RESAMPLES):
    selection=rng.integers(0,len(measured_score),len(measured_score))
    x_boot, y_boot=measured_score[selection], measured_log_dc50[selection]
    if np.ptp(x_boot)==0 or np.ptp(y_boot)==0: # degenerate resample, R2 undefined
        continue
    bootstrap_r2.append(stats.pearsonr(x_boot,y_boot).statistic**2)
r2_ci=np.percentile(bootstrap_r2,[2.5,97.5])
print(f'R2 of log10(DC50) vs NWHM on the {len(measured)} records with a quantitative DC50: {r2:.3f}, '
      f'95% CI [{r2_ci[0]:.3f}, {r2_ci[1]:.3f}] ({R2_BOOTSTRAP_RESAMPLES} paired bootstrap resamples, '
      f'seed {R2_BOOTSTRAP_SEED}, slope {slope:.2f})')

# the line spans the whole axes width and is clipped by the y limits, which follow the data
plotted_log_dc50=pedia_precompute['logDC50'].dropna()
ax.set_ylim(plotted_log_dc50.min()-0.2,plotted_log_dc50.max()+0.25)
x_limits=ax.get_xlim()
x_fit=np.linspace(x_limits[0],x_limits[1],50)
ax.plot(x_fit,intercept+slope*x_fit,color='0.25',linewidth=1.5,zorder=2)
ax.set_xlabel('NWHM',fontsize=13)
ax.set_ylabel('log(DC50/μM)',fontsize=13)
ax.tick_params(axis='both',labelsize=12)
# R2 sits in the legend frame as its title, so one box carries both the statistic and the marker key
ax.legend(fontsize=11,loc='lower left',framealpha=1,facecolor='white',alignment='left',
          title=f'$R^2$ = {r2:.2f} (n = {len(measured)}, 95% CI [{r2_ci[0]:.2f}, {r2_ci[1]:.2f}])',
          title_fontsize=11)
plt.tight_layout()
plt.savefig('../../pics/stat3_dc50_nwhm.png',dpi=600)
plt.close(fig)
