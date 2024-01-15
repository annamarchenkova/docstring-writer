import os
import yaml
import glob
from project_dirs import PROJECT_DIR, DATA_DIR


def load_config(cnf_dir=PROJECT_DIR, cnf_name='config.yml'):
    """
    load the yaml file
    """
    config_file = open(os.path.join(cnf_dir, cnf_name))
    return yaml.load(config_file, yaml.FullLoader)

def list_all_filepaths(
    common_dir=DATA_DIR,
    folder="",
    extension="txt",
):
    """_summary_

    Args:
        common_dir (_type_, optional): _description_. Defaults to DATA_DIR.
        folder (str, optional): _description_. Defaults to "".
        extension (str, optional): _description_. Defaults to "txt".

    Returns:
        _type_: _description_
    """
    path = os.path.join(common_dir, folder, f"*{extension}")
    filenames = glob.glob(path)

    if not filenames:
        path = os.path.join(
            common_dir, folder, "**\\", f"*{extension}"
        )  # search into subdirectories
        filenames = glob.glob(path)

    return filenames


DOCSTRING_FORMAT= """_summary_

    Args:
        arg1 (arg1_type): arg1_description. 
        argN (argN_type): argN_description. 

    Returns:
        _type_: _description_
    """

# ================== generative models ==================

def create_prompt(text):
    """Create prompt using text.

    Args:
        text (str): text to analyse.
    """

    prompt = (
        f"""Rewrite python function adding a docstring or updating existing docstring. {DOCSTRING_FORMAT}. Return a function with a docstring without any preamble text. Function to rewrite: {text}
    """.replace(
            "\n", ""
        )
        .replace("..", "")
        .strip()
    )
    
    return prompt
