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
pdbid_ripk1='4neu'
cmd.reinitialize()
cmd.fetch(pdbid_ripk1,name=pdbid_ripk1,type='cif',path='./')
ripk1_seq,ripk1_pocket=get_pocket_info_from_pdb_info(pdbid_ripk1,'Q1A','A','A','./',4)

# fetch MSA
get_a3m(ripk1_seq,pdbid_ripk1)
shutil.copy('../../data/MSA/crbn.a3m','crbn.a3m')
shutil.copy('../../data/MSA/vhl.a3m','vhl.a3m')
# get uniprot info
df_resi2evidence=update_uniprot_ptm_info('Q13546')
df_uid2seq=update_uniprot_uid2seq(['Q13546'])
df_uid2seq=df_uid2seq[df_uid2seq['UniprotID']=='Q13546']
seq_uniprot=df_uid2seq['seq'].iloc[0]
idxmap=idxmap_boltz2uniprot(ripk1_seq,seq_uniprot)

# calculate NWHM scores
pdb_df=pd.read_csv('ripk1_protac.csv',header='infer')
pdb_df['pdbid']=pdb_df['pdbid'].str.lower()
pdb_df['inchikey']=pdb_df['PROTAC SMILES'].apply(smi2inchikey)
pdb_df['seq_can']=ripk1_seq
pdb_df['boltz_pocket']=str(ripk1_pocket)
tmp_result=[]
for lig_smi,inchikey,name_poi,seq_poi,pocket_poi,name_e3 in pdb_df[['PROTAC SMILES','inchikey','pdbid','seq_can','boltz_pocket','E3 Ligase']].itertuples(False):
    result=calculate_one_affinity_score(lig_smi,name_poi,seq_poi,literal_eval(pocket_poi),name_e3,output_path='./',msa_path='./',extract_from_precomputed=False)
    tmp_result.append(result[1]|{'score':result[0]})
sc_df=pd.DataFrame(tmp_result)
pedia_precompute=pd.concat([pdb_df,sc_df],axis=1)

# annotate uniprot lys sites; NaN results are kept (and reported) instead of being dropped
pedia_precompute.rename(columns={'closest_lys_resid':'lys_site_boltz'},inplace=True)
uniprot_id='Q13546'
annotated=pedia_precompute['lys_site_boltz'].apply(lambda lys_boltz: map_lys_boltz2uniprot_evidence(lys_boltz,idxmap,df_resi2evidence,uniprot_id))
pedia_precompute['lys_site_uniprot']=[site for site,_ in annotated]
pedia_precompute['uniprot_lys_resid']=pedia_precompute['lys_site_uniprot']
pedia_precompute['evidence_score']=[score for _,score in annotated]
pedia_precompute['UniprotID']=uniprot_id

# estimate DC50 from the Hill equation; RIPK1 PROTACs only have a single degradation point at 1 uM
DC50_ASSAY_CONCENTRATION_UM=1.0
DC50_MAX_DEGRADATION=100.0
DC50_HILL_SLOPE=1.0
DEGRADATION_PATTERN=r'\s*([<>]?)\s*([0-9]*\.?[0-9]+)'


def parse_degradation(value):
    """Split '9.3' / '<5' into the numeric value and whether it is censored by the detection limit."""
    match=re.match(DEGRADATION_PATTERN,str(value))
    if match is None:
        return np.nan,True
    return float(match.group(2)),bool(match.group(1))


degradation_parsed=[parse_degradation(value) for value in pedia_precompute['RIPK1 degradation% @1μM']]
pedia_precompute['degradation_pct']=[value for value,_ in degradation_parsed]
pedia_precompute['degradation_censored']=[censored for _,censored in degradation_parsed]
# deg(C) = Dmax * C**h / (DC50**h + C**h)  ->  DC50 = C * (Dmax/deg - 1)**(1/h)
dc50_ratio=DC50_MAX_DEGRADATION/pedia_precompute['degradation_pct']-1
pedia_precompute['DC50_est_uM']=DC50_ASSAY_CONCENTRATION_UM*np.power(dc50_ratio.where(dc50_ratio>0),1/DC50_HILL_SLOPE)
pedia_precompute['logDC50_est']=np.log10(pedia_precompute['DC50_est_uM'])
print(f'estimated DC50 from the Hill equation (Dmax={DC50_MAX_DEGRADATION:g}%, slope={DC50_HILL_SLOPE:g}, '
      f'assay concentration={DC50_ASSAY_CONCENTRATION_UM:g} uM); records reported as "<5" stay at the detection '
      f'limit value and their DC50 is a lower bound')

pedia_precompute.to_csv('ripk1_precompute.csv',index=False)

# generate pse files; skip records whose lys site cannot be mapped to uniprot (e.g. FLAG/gene-edited)
for name_poi,name_e3,inchikey,lys_site_uniprot in pedia_precompute[['pdbid','E3 Ligase','inchikey','lys_site_uniprot']].itertuples(False):
    if pd.isna(lys_site_uniprot):
        logging.warning(f'Skip pse for {name_poi}_{name_e3}_{inchikey}: lys site cannot be mapped to uniprot (FLAG/gene editing).')
        continue
    if name_e3=='crbn':
        ref_e3_cif='../../data/PDB/clean/8rqa_ref.cif'
    else:
        ref_e3_cif='../../data/PDB/clean/6gfy_ref.cif'
    best_cif=f'{name_poi}_{name_e3}_{inchikey}_ternary_best.cif'
    cif_boltz2uniprot(best_cif,ripk1_seq,seq_uniprot)
    generate_pymol_pse(best_cif[:-4]+'_uni.cif',lys_site_uniprot,'clean/4neu_ref.cif','',ref_e3_cif)

# NWHM vs estimated log(DC50); marker shape and colour encode the E3 ligase, filled/hollow encodes data availability
E3_STYLE={'crbn':('o','tab:red','CRBN'),'vhl':('s','tab:blue','VHL')}
fig,ax=plt.subplots(figsize=(5,5))
for e3,(marker,color,e3_label) in E3_STYLE.items():
    for censored,quality in [(False,'>=5% at 1 μM'),(True,'<5% at 1 μM')]:
        subset=pedia_precompute[(pedia_precompute['E3 Ligase'].str.lower()==e3)&(pedia_precompute['degradation_censored']==censored)]
        if subset.empty:
            continue
        ax.scatter(subset['score'],subset['logDC50_est'],s=45,marker=marker,zorder=3,
                   facecolors='none' if censored else color,edgecolors=color,
                   label=f'{e3_label}, {quality} (n = {len(subset)})')
for level in pedia_precompute.loc[pedia_precompute['degradation_censored'],'logDC50_est'].dropna().unique():
    ax.axhline(level,color='gray',linestyle='--',linewidth=1.2) # detection limit of the single-point assay

# R2 and its interval use only the records with a measured degradation value
R2_BOOTSTRAP_RESAMPLES=2000
R2_BOOTSTRAP_SEED=0
measured=pedia_precompute[~pedia_precompute['degradation_censored']].dropna(subset=['score','logDC50_est'])
measured_score=measured['score'].to_numpy(dtype=float)
measured_log_dc50=measured['logDC50_est'].to_numpy(dtype=float)
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
print(f'R2 of log10(DC50) vs NWHM on the {len(measured)} measured records: {r2:.3f}, '
      f'95% CI [{r2_ci[0]:.3f}, {r2_ci[1]:.3f}] ({R2_BOOTSTRAP_RESAMPLES} paired bootstrap resamples, '
      f'seed {R2_BOOTSTRAP_SEED}, slope {slope:.2f})')

# the line spans the whole axes width and is clipped by the y limits, which follow the data
plotted_log_dc50=pedia_precompute['logDC50_est'].dropna()
ax.set_ylim(plotted_log_dc50.min()-0.2,plotted_log_dc50.max()+0.25)
x_limits=ax.get_xlim()
x_fit=np.linspace(x_limits[0],x_limits[1],50)
ax.plot(x_fit,intercept+slope*x_fit,color='0.25',linewidth=1.5,zorder=2)
ax.set_xlabel('NWHM',fontsize=11)
ax.set_ylabel('log(DC50/μM)',fontsize=11)
# R2 sits in the legend frame as its title, so one box carries both the statistic and the marker key
ax.legend(fontsize=9,loc='lower left',framealpha=1,facecolor='white',alignment='left',
          title=f'$R^2$ = {r2:.2f} (n = {len(measured)}, 95% CI [{r2_ci[0]:.2f}, {r2_ci[1]:.2f}])',
          title_fontsize=9)
plt.tight_layout()
plt.savefig('../../pics/ripk1_dc50_nwhm.png',dpi=600)
plt.close(fig)