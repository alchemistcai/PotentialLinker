from rdkit.Chem import MolFromSmiles,MolToInchiKey

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