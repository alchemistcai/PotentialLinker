# PotentialLinker, a PROTAC linker generator in a linearly additive energy space

![](fgfr4_d11.png)

## Prepare envs

We recommand CUDA GPU with 32G VRAM. 11G lead to ~1/2 complex out of memory, only suitable for small POI with VHL.
On RTX4080 32G, a ternary spends about 2 mins. One RTX4060 8G, it will takes 10 mins.

REINVENT4 and boltz supports CPU devices. However, it will be too slow for boltz.

```bash
os.environ['BOLTZ_CACHE']='/root/autodl-tmp/boltz'
mamba create -n boltz python=3.11
mamba activate boltz
git clone https://github.com/MolecularAI/REINVENT4.git --depth 1
cd REINVENT4
python install.py cu126
cd ..
git clone xxxx boltz
cd boltz
pip install -e .[cuda]
cd ..
git clone xxx
cd potential_linker
pip install xxx
# if you want to use precomputed results directly
wget zenodoxxxxre
tar xxxx


mamba install pymol-open-source ipykernel biopython numpy=1.26.4 scikit-learn=1.6.1
```

If you want to precomputed results from raw database, execute the `ipynb` notebook and generating the results(about 30h).

Using different GPU may lead to very slight difference because of device precision.

If you fail to install REINVENT4 because of ISIM, alter the dependencies `REINVENT4/pyproject.toml` to my mirror repo:

```toml
isim = [
  "iSIM@git+https://gitee.com/regentsai/iSIM.git"
]
```

## Train REINVENT agents for interested targets

- Prepare a smi file including POI and E3 part ligands, split by `|`. The connect points need to be marked as `*`. 
  - CRBN-LVY / VHL-EXH E3 part is already in `boltz_fgfr4_vhl|crbn_warhead.smi.`
  - `*` markers don't need to be the first/last character.
  - You can use [Zinc15](https://zinc15.docking.org/substances/home/), ChemDraw or other softwares/webs to draw and get their smiles. Visit [CCD](https://www.ebi.ac.uk/pdbe-srv/pdbechem/) according to PDB ligand CCD code can get an unmarked SMILES too.
  - Use a unique atom like P/Br/Pt as a placeholder and alter it to `*` manually later if your software don't support wildcards.
  - Covalent ligands should remove leaving groups. Addition reaction sites should be added already without protein sidechains.
    - For example, michael receptor "C=C-C(=O)xxxx-linker" should be `C-C-C(=O)xxxx*`.
    - Chloroacetamide "ClC(=O)NH-xxxx-linker" should be `C(=O)NH-xxxx*`.
    - Again, CCD gives an reacted product SMILES if it is a covalent ligand.
- Modify `boltz_fgfr4_vhl|crbn.toml` and adjust parameters in `[[stage.scoring.component.BoltzScore.endpoint]]`.
  - Log name, checkpoint names and other related parameters should be altered too.
  - `covalent_xxx` parameters are only required for covalent PROTACs (See fgfr4 examples). 
  - `params.lig_warhead_smarts` is the covalent warhead part after reaction. `[C:1]-C-C=O` is the added warhead and [X:1] is used to mark the covalent connection point.

## Sampling PROTACs

- asdfs
- In Reinvent, sampling does not calculate scoring. You need to calculate them by yourself.
- Use xxxx

## Our Boltz Modification

- Enable predicting affinity for ligands with more than 56 atoms.
- Atom name method is not changed because it effects affinity scores calculation. "CL1xx","BR1xx" with more than 4 characters are treated as bad atoms right now.
- You can clone Boltz official repo and alter it by yourself to keep update with it.

## Cite Us

xxxx
