import zipfile
import pandas as pd


def list_files(zip_path: str) -> list:
    with zipfile.ZipFile(zip_path) as z:
        return [f for f in z.namelist() if not f.endswith('/')]


def load_file(zip_path: str, filename: str):
    """
    Load one file from zip.
    .csv  -> DataFrame
    .xlsx -> dict of {sheet_name: DataFrame}
    Handles utf-8 and latin1 encoding automatically.
    """
    with zipfile.ZipFile(zip_path) as z:
        with z.open(filename) as f:
            if filename.endswith('.xlsx'):
                xl = pd.ExcelFile(f)
                return {sheet: xl.parse(sheet) for sheet in xl.sheet_names}
            try:
                return pd.read_csv(f, encoding='utf-8', low_memory=False)
            except UnicodeDecodeError:
                pass
    with zipfile.ZipFile(zip_path) as z:
        with z.open(filename) as f:
            return pd.read_csv(f, encoding='latin1', low_memory=False)
