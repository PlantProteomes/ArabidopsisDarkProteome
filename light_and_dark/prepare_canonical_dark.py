#!/bin/env python3

import re
from Bio import SeqIO
import Bio.SeqUtils.ProtParam
import csv

from attr import define


filename = 'Araport11.fasta'
proteins = {}
n_entries = 0
with open(filename) as infile:
    for record in SeqIO.parse(infile, 'fasta'):

        # parse line and separate entry into identifier and description
        match = re.match(r'^(\S+)\s*(.*)$', record.description)

        # if identifiers are parseable store appropriate values. Else print error statement
        if match:
            identifier = match.group(1)
            description = match.group(2)
            sequence = str(record.seq)
            temp_sequence = sequence.replace('X', 'L')
            temp_sequence = temp_sequence.replace('*', 'L')
            protein_analysis = Bio.SeqUtils.ProtParam.ProteinAnalysis(str(temp_sequence))

            proteins[identifier] = {
                'description': description,
                'sequence': sequence,
                'molecular_weight': protein_analysis.molecular_weight(),
                'gravy': protein_analysis.gravy(),
                'isoelectric_point': protein_analysis.isoelectric_point()
            }


        else:
            print(
                f"ERROR: Unable to parse description line: {record.description}")
            exit()
        n_entries += 1
print(f"INFO: Read {n_entries} entries from {filename}")


rna_metrics = {}
filename = 'Expression Metrics--All Genes.tsv'
print(f"INFO: Reading {filename}")
with open(filename) as infile:
    lines = csv.reader(infile, delimiter="\t")
    row_counter = 0
    for columns in lines:
        row_counter += 1
        if row_counter == 1:
            continue
        gene_identifier = columns[0]
        detected_percent = columns[3]
        highest_tpm = columns[9]
        rna_metrics[gene_identifier] = [ detected_percent, highest_tpm ]



canonical_list_file = open('canonical_list.tsv', 'w')
canonical_list = []

not_detected_list_file = open('not_detected_list.tsv', 'w')
not_detected_list = []

outfile = open('light_and_dark_protein_list.tsv', 'w')
print("\t".join(['identifier', 'gene_symbol', 'chromosome', 'status', 'n_obs', 'molecular_weight', 'gravy', 'pI', 'rna_detected_percent', 'highest_tpm', 'description']), file=outfile)
light_and_dark_protein_list = []

filename = 'query_guest_20230423-205433.tsv'
print(f"INFO: Reading {filename}")
with open(filename) as infile:
    lines = csv.reader(infile, delimiter="\t")
    row_counter = 0
    n_skipped = 0
    for columns in lines:
        row_counter += 1
        if row_counter == 1:
            continue
        gene_symbol= columns[0]
        identifier = columns[1]
        chromosome = columns[6]
        status = columns[11]
        n_obs = columns[12]
        description = columns[13].lstrip().rstrip()

        if identifier.startswith('PeptideAtlas'):
            continue

        if status == 'canonical':
            canonical_list.append(identifier)
        elif status == 'not observed':
            not_detected_list.append(identifier)
        else:
            #print(f"ERROR: Unknown status {status}")
            #exit()
            pass

        molecular_weight = str(0.0)
        if identifier in proteins:
            molecular_weight = str(proteins[identifier]['molecular_weight'] / 1000)
        gravy = str(proteins[identifier]['gravy'])
        pI = str(proteins[identifier]['isoelectric_point'])

        rna_detected_percent = str(0.0)
        highest_tpm = str(0.0)
        gene_identifier = identifier[0:len(identifier)-2]
        if gene_identifier in rna_metrics:
            rna_detected_percent = rna_metrics[gene_identifier][0]
            highest_tpm = rna_metrics[gene_identifier][1]
        else:
            print(f"WARNING: No RNA data for {gene_identifier}. Skipping")
            n_skipped += 1
            continue

        row = [identifier, gene_symbol, chromosome, status, n_obs, molecular_weight, gravy, pI, rna_detected_percent, highest_tpm, description]
        light_and_dark_protein_list.append(row)

print(f"WARNING: Skipped {n_skipped} proteins with no corresponding RNA information")

for identifier in sorted(canonical_list):
    print(identifier, file=canonical_list_file)

for identifier in sorted(not_detected_list):
    print(identifier, file=not_detected_list_file)

sorted_list = sorted(light_and_dark_protein_list, key=lambda x:x[3]+x[0])
for row in sorted_list:
    print("\t".join(row), file=outfile)

exit()



