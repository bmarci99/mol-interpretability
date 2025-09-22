# Robust RDKit FilterCatalog access, works across builds.

from rdkit.Chem import FilterCatalog as FC

def available_catalogs():
    enums = FC.FilterCatalogParams.FilterCatalogs
    names = [n for n in dir(enums) if n.isupper() and hasattr(enums, n)]
    return names

def rdkit_alerts_smarts(include=None):
    """
    Return {label -> SMARTS} for the chosen catalogs.
    - include=None : add ALL if present, else add every present catalog.
    - include=list[str] : add just those (if present). E.g. ["PAINS_A","PAINS_B","PAINS_C","BRENK","NIH","ZINC"]
    """
    enums = FC.FilterCatalogParams.FilterCatalogs
    params = FC.FilterCatalogParams()

    if include is None:
        if hasattr(enums, "ALL"):
            params.AddCatalog(enums.ALL)
        else:
            for name in available_catalogs():
                params.AddCatalog(getattr(enums, name))
    else:
        for name in include:
            if hasattr(enums, name):
                params.AddCatalog(getattr(enums, name))

    cat = FC.FilterCatalog(params)
    N = cat.GetNumEntries()
    if N == 0:
        return {}

    smarts = {}
    for i in range(N):
        e = cat.GetEntryWithIdx(i)
        # Always available: description + SMARTS
        desc = e.GetDescription() if hasattr(e, "GetDescription") else f"entry_{i}"
        s    = e.GetSmarts() if hasattr(e, "GetSmarts") else None
        if s:
            # Unique, readable label without relying on GetCatalogName()
            smarts[f"RDKit:{i}:{desc}"] = s
    return smarts
