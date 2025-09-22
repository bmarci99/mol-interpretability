# llm_mol_interp/utils/rdkit_logs.py
from rdkit import RDLogger

def silence(warnings: bool = True, errors: bool = False):
    if warnings:
        RDLogger.DisableLog('rdApp.warning')
    if errors:
        RDLogger.DisableLog('rdApp.error')