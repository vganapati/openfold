import numpy as np
import gemmi
import torch

def openfold_to_gemmi(sequence, positions, mask, b_factors):
    """
    Convert OpenFold output to a gemmi.Structure object.
    
    :param sequence: String of amino acid sequence
    :param positions: Tensor of atom positions (N, CA, C, O for each residue)
    :param mask: Tensor indicating which positions are valid
    :param b_factors: Tensor of B-factors for each residue
    :return: gemmi.Structure object
    """
    # Create a new gemmi Structure
    structure = gemmi.Structure()
    model = gemmi.Model("1")
    chain = gemmi.Chain("A")

    # Mapping from one-letter to three-letter amino acid codes
    aa_map = {
        'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
        'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
        'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
        'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
    }

    # Convert tensors to numpy arrays
    positions = positions.cpu().numpy()
    mask = mask.cpu().numpy()
    b_factors = b_factors.cpu().numpy()

    for i, (aa, pos, m, b) in enumerate(zip(sequence, positions, mask, b_factors)):
        if m:  # Only add residues that are marked as valid in the mask
            residue = gemmi.Residue()
            residue.name = aa_map[aa]
            residue.seqid = gemmi.SeqId(i + 1, ' ')

            # Add backbone atoms
            for atom_name, coords in zip(['N', 'CA', 'C', 'O'], pos):
                atom = gemmi.Atom()
                atom.name = atom_name
                atom.pos = gemmi.Position(*coords)
                atom.b_iso = b
                atom.occ = 1.0
                atom.element = gemmi.Element(atom_name[0])
                residue.add_atom(atom)

            chain.add_residue(residue)

    model.add_chain(chain)
    structure.add_model(model)

    return structure

# Example usage (you'll need to replace these with actual OpenFold outputs)
sequence = "SEQUENCE"  # Replace with actual sequence
positions = torch.rand(len(sequence), 4, 3)  # Replace with actual positions
mask = torch.ones(len(sequence), dtype=torch.bool)  # Replace with actual mask
b_factors = torch.rand(len(sequence))  # Replace with actual B-factors

structure = openfold_to_gemmi(sequence, positions, mask, b_factors)

# Save the structure to a PDB file
structure.write_pdb("openfold_output.pdb")