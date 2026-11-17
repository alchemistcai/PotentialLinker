# PotentialLinker 1.1.0

2026.11.17

- An optional environment yaml for version reference.
- E3 ligases now have uniprot residue ids too. Related cif and pse files were updated.
- Support covalent warheads for E3 now.
- Create an example notebook in benchmark/STAT3: using predicted STAT3-27(SD-965)-CRBN ternary as a lead template, for further optimization.
- More user-friendly experiment constants in FGFR4 notebook. Users can input known uniprot residue ids for covalent sites, PPI and pockets.
  - For example, one does not need to confirm that C552, the covalent residue of FGFR4, is numbered 104 when the sequence starts counting from 1 in PDB 4XCU.
  - Just input 552 and give the PDB sequence.
